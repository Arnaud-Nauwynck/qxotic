package com.qxotic.jota.runtime.iree;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Indexing;
import com.qxotic.jota.Shape;
import com.qxotic.jota.ir.tir.ScalarInput;
import com.qxotic.jota.ir.tir.TIRGraph;
import com.qxotic.jota.ir.tir.TIRNode;
import com.qxotic.jota.ir.tir.TensorInput;
import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.runtime.KernelCachePaths;
import com.qxotic.jota.runtime.iree.compiler.IreeCompiler;
import com.qxotic.jota.runtime.iree.runtime.HostTensor;
import com.qxotic.jota.runtime.iree.runtime.IreeDevice;
import com.qxotic.jota.runtime.iree.runtime.IreeElementType;
import com.qxotic.jota.runtime.iree.runtime.IreeModule;
import com.qxotic.jota.runtime.iree.runtime.IreeRuntimeException;
import com.qxotic.jota.runtime.iree.runtime.IreeRuntimeLibrary;
import com.qxotic.jota.stablehlo.ir.Program;
import com.qxotic.jota.stablehlo.portable.PortableArtifact;
import com.qxotic.jota.stablehlo.portable.StableHloVersion;
import com.qxotic.jota.stablehlo.text.MlirPrinter;
import com.qxotic.jota.stablehlo.tir.TirToStableHlo;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Runs jota {@link TIRGraph}s on IREE: export to StableHLO ({@link TirToStableHlo}), compile to a
 * {@code .vmfb} (cached in memory and on disk), load on the device, copy inputs in and outputs out.
 *
 * <p>Phase A of the plan (host memory): inputs are host {@link MemorySegment} views, strided ones
 * gathered to row-major first; outputs land in a host {@link MemoryDomain}.
 */
public final class IreeGraphExecutor implements AutoCloseable {

    /** How programs are handed to the compiler. */
    public enum Format {
        /** MLIR text: version-agnostic and readable in dumps; the default. */
        TEXT,
        /** A VHLO portable artifact for {@link IreeConfig#STABLEHLO_VERSION}. */
        PORTABLE;

        public static Format parse(String text) {
            try {
                return valueOf(text.strip().toUpperCase(Locale.ROOT));
            } catch (IllegalArgumentException e) {
                throw new IllegalArgumentException(
                        "unknown IREE program format '" + text + "' (expected text or portable)");
            }
        }
    }

    private final IreeCompiler compiler;
    private final IreeDevice device;
    private final Format format;
    private final StableHloVersion portableVersion;
    private final Optional<Path> diskCache;
    private final Map<String, IreeModule> loaded = new ConcurrentHashMap<>();
    private final java.util.concurrent.atomic.AtomicInteger compiles =
            new java.util.concurrent.atomic.AtomicInteger();

    public IreeGraphExecutor(
            IreeCompiler compiler,
            IreeDevice device,
            Format format,
            StableHloVersion portableVersion,
            Optional<Path> diskCache) {
        this.compiler = Objects.requireNonNull(compiler, "compiler");
        this.device = Objects.requireNonNull(device, "device");
        this.format = Objects.requireNonNull(format, "format");
        this.portableVersion = Objects.requireNonNull(portableVersion, "portableVersion");
        this.diskCache = Objects.requireNonNull(diskCache, "diskCache");
    }

    /**
     * An executor from the configured compiler, runtime library and driver; the disk cache is
     * {@code <KernelCachePaths.versionRoot()>/iree}.
     */
    public static IreeGraphExecutor fromConfig(IreeConfig config) {
        IreeCompiler compiler = IreeCompiler.fromConfig(config);
        Path runtime =
                config.runtimeLibrary()
                        .orElseThrow(
                                () ->
                                        new IreeRuntimeException(
                                                "No IREE runtime library configured: "
                                                        + IreeConfig.runtimeInstallHint()));
        IreeDevice device = IreeDevice.create(IreeRuntimeLibrary.open(runtime), config.driver());
        return new IreeGraphExecutor(
                compiler,
                device,
                config.format(),
                config.stableHloVersion(),
                Optional.of(KernelCachePaths.versionRoot().resolve("iree")));
    }

    /** {@code ireeCompilerGetRevision()} of the compiler in use. */
    public String compilerRevision() {
        return compiler.library().revision();
    }

    /** Number of programs compiled (cache misses in memory and on disk) by this executor. */
    public int compileCount() {
        return compiles.get();
    }

    /**
     * Runs {@code graph} with one view per {@link TIRGraph#inputs()} input, returning one row-major
     * view per output, allocated in {@code outputDomain}.
     */
    public List<MemoryView<MemorySegment>> execute(
            TIRGraph graph,
            List<? extends MemoryView<MemorySegment>> inputs,
            MemoryDomain<MemorySegment> outputDomain) {
        if (inputs.size() != graph.inputCount()) {
            throw new IllegalArgumentException(
                    "graph has " + graph.inputCount() + " inputs, got " + inputs.size());
        }
        IreeModule module = moduleFor(graph);
        try (Arena arena = Arena.ofConfined()) {
            var args = new ArrayList<HostTensor>(inputs.size());
            for (int i = 0; i < inputs.size(); i++) {
                args.add(argument(graph.inputs().get(i), inputs.get(i), arena));
            }
            List<HostTensor> results = module.call(IreeModule.MAIN, args);
            if (results.size() != graph.outputCount()) {
                throw new IreeRuntimeException(
                        "IREE returned "
                                + results.size()
                                + " outputs, expected "
                                + graph.outputCount());
            }
            var outputs = new ArrayList<MemoryView<MemorySegment>>(results.size());
            for (int k = 0; k < results.size(); k++) {
                outputs.add(output(graph.outputs().get(k), results.get(k), outputDomain));
            }
            return outputs;
        }
    }

    /** The StableHLO bytes sent to the compiler for {@code graph}. */
    public byte[] encode(TIRGraph graph) {
        Program program = TirToStableHlo.export(graph);
        return switch (format) {
            case TEXT -> MlirPrinter.print(program).getBytes(StandardCharsets.UTF_8);
            case PORTABLE -> PortableArtifact.serialize(program, portableVersion);
        };
    }

    /**
     * Compiles (or loads from the disk cache) and loads {@code graph}'s module, without running it.
     */
    public void prepare(TIRGraph graph) {
        moduleFor(graph);
    }

    private IreeModule moduleFor(TIRGraph graph) {
        byte[] source = encode(graph);
        String key = cacheKey(source);
        IreeModule module = loaded.get(key);
        if (module != null) {
            return module;
        }
        // compile outside any lock: concurrent precompiles of different graphs overlap
        byte[] vmfb = vmfb(key, source);
        synchronized (this) {
            module = loaded.get(key);
            if (module == null) {
                module = device.load(vmfb);
                loaded.put(key, module);
            }
            return module;
        }
    }

    private byte[] vmfb(String key, byte[] source) {
        Optional<Path> file = diskCache.map(dir -> dir.resolve(key + ".vmfb"));
        if (file.isPresent() && Files.isRegularFile(file.get())) {
            try {
                return Files.readAllBytes(file.get());
            } catch (IOException e) {
                // unreadable cache entry: recompile and overwrite
            }
        }
        byte[] vmfb = compiler.compile(source).vmfb();
        compiles.incrementAndGet();
        if (file.isPresent()) {
            write(file.get(), vmfb);
        }
        return vmfb;
    }

    private String cacheKey(byte[] source) {
        try {
            MessageDigest sha = MessageDigest.getInstance("SHA-256");
            sha.update(source);
            sha.update((byte) 0);
            sha.update(String.join("\u0000", compiler.flags()).getBytes(StandardCharsets.UTF_8));
            sha.update((byte) 0);
            sha.update(compiler.library().revision().getBytes(StandardCharsets.UTF_8));
            return HexFormat.of().formatHex(sha.digest(), 0, 16);
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException(e);
        }
    }

    private static void write(Path file, byte[] bytes) {
        try {
            Files.createDirectories(file.getParent());
            Path tmp =
                    Files.createTempFile(file.getParent(), file.getFileName().toString(), ".tmp");
            Files.write(tmp, bytes);
            Files.move(
                    tmp, file, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        } catch (IOException e) {
            throw new UncheckedIOException("Cannot write IREE module cache " + file, e);
        }
    }

    private static HostTensor argument(TIRNode input, MemoryView<MemorySegment> view, Arena arena) {
        DataType dt = input.dataType();
        if (view.dataType() != dt) {
            throw new IllegalArgumentException(
                    "input " + input + " is " + dt + ", got a " + view.dataType() + " view");
        }
        int elementType = IreeElementType.of(dt);
        return switch (input) {
            // exported as a rank-0 tensor and broadcast inside the program
            case ScalarInput si -> new HostTensor(elementType, new long[0], gather(view, 1, arena));
            case TensorInput ti -> {
                if (!view.layout().shape().equals(ti.shape())) {
                    throw new IllegalArgumentException(
                            "input "
                                    + ti
                                    + " has shape "
                                    + ti.shape()
                                    + ", got "
                                    + view.layout().shape());
                }
                Shape shape = ti.shape();
                yield new HostTensor(
                        elementType,
                        shape.flattenModes().toArray(),
                        gather(view, shape.size(), arena));
            }
            default -> throw new IllegalArgumentException("not a graph input: " + input);
        };
    }

    /**
     * The first {@code elements} elements of {@code view} in row-major order: a slice of its memory
     * when already dense row-major, else a copy.
     */
    static MemorySegment gather(MemoryView<MemorySegment> view, long elements, Arena arena) {
        long bytes = view.dataType().byteSize();
        MemorySegment base = view.memory().base();
        if (view.isRowMajorContiguous()) {
            return base.asSlice(view.byteOffset(), elements * bytes);
        }
        MemorySegment out = arena.allocate(elements * bytes, 64);
        for (long i = 0; i < elements; i++) {
            // element strides, absolute byte offset (includes view.byteOffset())
            MemorySegment.copy(base, Indexing.linearToOffset(view, i), out, i * bytes, bytes);
        }
        return out;
    }

    private static MemoryView<MemorySegment> output(
            TIRNode node, HostTensor result, MemoryDomain<MemorySegment> domain) {
        DataType dt = node.dataType();
        if (!IreeElementType.compatible(result.elementType(), dt)) {
            throw new IreeRuntimeException(
                    "IREE output "
                            + IreeElementType.describe(result.elementType())
                            + " does not match "
                            + dt);
        }
        Shape shape = node.shape();
        if (shape.size() * dt.byteSize() != result.data().byteSize()) {
            throw new IreeRuntimeException(
                    "IREE output of "
                            + result.data().byteSize()
                            + " bytes does not match "
                            + dt
                            + shape);
        }
        Memory<MemorySegment> memory = domain.memoryAllocator().allocateMemory(dt, shape);
        MemorySegment.copy(result.data(), 0, memory.base(), 0, result.data().byteSize());
        if (dt == DataType.BOOL) {
            MemorySegment seg = memory.base();
            for (long i = 0; i < seg.byteSize(); i++) {
                seg.set(
                        ValueLayout.JAVA_BYTE,
                        i,
                        (byte) (seg.get(ValueLayout.JAVA_BYTE, i) != 0 ? 1 : 0));
            }
        }
        return MemoryView.rowMajor(memory, dt, shape);
    }

    @Override
    public void close() {
        for (IreeModule m : loaded.values()) {
            m.close();
        }
        loaded.clear();
        device.close();
    }
}
