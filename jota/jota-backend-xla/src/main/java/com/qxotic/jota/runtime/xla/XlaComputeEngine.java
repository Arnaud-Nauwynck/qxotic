package com.qxotic.jota.runtime.xla;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Device;
import com.qxotic.jota.Environment;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import com.qxotic.jota.ir.tir.ScalarInput;
import com.qxotic.jota.ir.tir.TIRGraph;
import com.qxotic.jota.ir.tir.TIRNode;
import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.runtime.ComputeEngine;
import com.qxotic.jota.runtime.xla.pjrt.CompileOptions;
import com.qxotic.jota.runtime.xla.pjrt.PjrtBuffer;
import com.qxotic.jota.runtime.xla.pjrt.PjrtDevice;
import com.qxotic.jota.runtime.xla.pjrt.PjrtLoadedExecutable;
import com.qxotic.jota.stablehlo.ir.Program;
import com.qxotic.jota.stablehlo.text.MlirPrinter;
import com.qxotic.jota.stablehlo.tir.TirToStableHlo;
import com.qxotic.jota.tensor.Tensor;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.MemorySegment;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Executes TIR graphs on one PJRT device: export to StableHLO, compile (cached by the hash of the
 * encoded program), copy the inputs to the device, execute, copy the result back into this device's
 * host memory.
 */
final class XlaComputeEngine implements ComputeEngine {

    private final XlaBackend backend;
    private final XlaMemoryDomain memoryDomain;
    private final PjrtDevice pjrtDevice;
    private final byte[] compileOptions;
    private final Map<String, PjrtLoadedExecutable> executables;

    XlaComputeEngine(XlaBackend backend, XlaMemoryDomain memoryDomain, PjrtDevice pjrtDevice) {
        this.backend = Objects.requireNonNull(backend, "backend");
        this.memoryDomain = Objects.requireNonNull(memoryDomain, "memoryDomain");
        this.pjrtDevice = Objects.requireNonNull(pjrtDevice, "pjrtDevice");
        this.compileOptions = CompileOptions.singleDevice(pjrtDevice.id());
        int capacity = backend.cacheSize();
        this.executables =
                new LinkedHashMap<>(16, 0.75f, true) {
                    @Override
                    protected boolean removeEldestEntry(
                            Map.Entry<String, PjrtLoadedExecutable> eldest) {
                        if (size() > capacity) {
                            eldest.getValue().close();
                            return true;
                        }
                        return false;
                    }
                };
    }

    @Override
    public Device device() {
        return memoryDomain.device();
    }

    @Override
    public boolean supportsParallelPrecompile() {
        return true;
    }

    @Override
    public void precompile(TIRGraph graph) {
        executable(graph);
    }

    @Override
    public MemoryView<?> execute(TIRGraph graph, List<Tensor> inputs) {
        if (graph.inputs().size() != inputs.size()) {
            throw new IllegalArgumentException(
                    "Expected " + graph.inputs().size() + " inputs but got " + inputs.size());
        }
        TIRNode output = graph.outputs().getFirst();
        XlaDataTypes.of(output.dataType());
        PjrtLoadedExecutable exe = executable(graph);

        List<PjrtBuffer> arguments = new ArrayList<>(inputs.size());
        List<PjrtBuffer> owned = new ArrayList<>(inputs.size());
        List<PjrtBuffer> results = new ArrayList<>();
        XlaMemory.LOCK.lock();
        try {
            for (int i = 0; i < inputs.size(); i++) {
                arguments.add(argument(graph.inputs().get(i), inputs.get(i), owned));
            }
            results.addAll(exe.execute(arguments));
            PjrtBuffer result = results.removeFirst();
            return keepOnDevice(result, output.dataType(), output.shape());
        } finally {
            XlaMemory.LOCK.unlock();
            owned.forEach(PjrtBuffer::close);
            results.forEach(PjrtBuffer::close);
        }
    }

    // ------------------------------------------------------------- compilation

    private PjrtLoadedExecutable executable(TIRGraph graph) {
        Program program = TirToStableHlo.export(graph);
        byte[] code = XlaPrograms.encode(program, backend.format(), backend.target());
        String key = backend.format() + "-" + sha256(code);
        synchronized (executables) {
            PjrtLoadedExecutable exe = executables.get(key);
            if (exe != null) {
                return exe;
            }
        }
        backend.dumpDir().ifPresent(dir -> dump(dir, key, program, code));
        PjrtLoadedExecutable loaded = loadOrCompile(key + "-d" + pjrtDevice.id(), code);
        synchronized (executables) {
            PjrtLoadedExecutable raced = executables.putIfAbsent(key, loaded);
            if (raced != null) {
                loaded.close();
                return raced;
            }
            return loaded;
        }
    }

    /** From the disk cache when it has a loadable entry, otherwise compiled (and stored). */
    private PjrtLoadedExecutable loadOrCompile(String diskKey, byte[] code) {
        XlaExecutableStore store = backend.store();
        var serialized = store.load(diskKey);
        if (serialized.isPresent()) {
            try {
                return backend.client().deserialize(serialized.get(), compileOptions);
            } catch (RuntimeException e) {
                store.delete(diskKey);
            }
        }
        PjrtLoadedExecutable compiled =
                backend.client().compile(code, XlaPrograms.PJRT_FORMAT, compileOptions);
        if (store.isEnabled()) {
            try {
                store.store(diskKey, compiled.serialize());
            } catch (RuntimeException e) {
                store.disable("the plugin cannot serialize executables", e);
            }
        }
        return compiled;
    }

    private static void dump(Path dir, String key, Program program, byte[] code) {
        try {
            Files.createDirectories(dir);
            Files.writeString(dir.resolve(key + ".mlir"), MlirPrinter.print(program));
            Files.write(dir.resolve(key + ".bin"), code);
        } catch (IOException e) {
            throw new UncheckedIOException("Cannot dump XLA program to " + dir, e);
        }
    }

    private static String sha256(byte[] bytes) {
        try {
            return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes));
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException(e);
        }
    }

    // ---------------------------------------------------------------- transfers

    /**
     * The device buffer for one graph input: the device mirror of an {@code xla} memory the view
     * covers exactly, else a fresh upload (added to {@code owned}, closed after the execution); a
     * scalar input becomes a 0-d array.
     */
    private PjrtBuffer argument(TIRNode input, Tensor tensor, List<PjrtBuffer> owned) {
        MemoryView<?> view = tensor.materialize();
        DataType dt = input.dataType();
        if (view.dataType() != dt) {
            throw new IllegalArgumentException(
                    "input " + input + " expects " + dt + " but got " + view.dataType());
        }
        long[] dims =
                input instanceof ScalarInput ? new long[0] : input.shape().flattenModes().toArray();
        PjrtBuffer mirror = mirrorOf(view, dims);
        if (mirror != null) {
            XlaMemory.REUSES.incrementAndGet();
            return mirror;
        }
        MemorySegment host = hostRowMajor(view);
        if (input instanceof ScalarInput) {
            host = host.asSlice(0, dt.byteSize());
        }
        PjrtBuffer uploaded =
                backend.client().bufferFromHost(host, XlaDataTypes.of(dt), dims, pjrtDevice);
        owned.add(uploaded);
        XlaMemory.UPLOADS.incrementAndGet();
        return uploaded;
    }

    /** The device mirror holding exactly the view's elements as {@code dims}, or null. */
    private PjrtBuffer mirrorOf(MemoryView<?> view, long[] dims) {
        if (!(view.memory() instanceof XlaMemory memory)
                || view.byteOffset() != 0
                || !view.isRowMajorContiguous()
                || view.dataType().byteSizeFor(view.shape().size()) != memory.byteSize()) {
            return null;
        }
        return memory.deviceBuffer(dims, pjrtDevice.id());
    }

    /**
     * The view's elements as a dense row-major host segment: the view's own memory when it is
     * already host-addressable and row-major contiguous, otherwise a copy into this domain.
     */
    private MemorySegment hostRowMajor(MemoryView<?> view) {
        long bytes = view.dataType().byteSizeFor(view.shape().size());
        if (view.memory().base() instanceof MemorySegment segment && view.isRowMajorContiguous()) {
            return segment.asSlice(view.byteOffset(), bytes);
        }
        Memory<MemorySegment> memory =
                memoryDomain.memoryAllocator().allocateMemory(Math.max(bytes, 1));
        MemoryView<MemorySegment> dst =
                MemoryView.of(memory, 0, view.dataType(), Layout.rowMajor(view.shape()));
        copy(view, dst);
        return memory.base().asSlice(0, bytes);
    }

    @SuppressWarnings("unchecked")
    private void copy(MemoryView<?> src, MemoryView<MemorySegment> dst) {
        MemoryDomain<Object> srcDomain =
                (MemoryDomain<Object>) Environment.runtimeFor(src.memory().device()).memoryDomain();
        MemoryDomain.copy(srcDomain, (MemoryView<Object>) src, memoryDomain, dst);
    }

    /**
     * A new row-major view of this device's memory whose contents are {@code buffer}: the elements
     * stay on the device until something reads the host memory. Takes ownership of the buffer.
     */
    private MemoryView<MemorySegment> keepOnDevice(PjrtBuffer buffer, DataType dt, Shape shape) {
        if (buffer.type() != XlaDataTypes.of(dt)) {
            buffer.close();
            throw new IllegalStateException(
                    "XLA returned " + buffer.type() + " where " + dt + " was expected");
        }
        long bytes = dt.byteSizeFor(shape.size());
        XlaMemory memory =
                (XlaMemory) memoryDomain.memoryAllocator().allocateMemory(Math.max(bytes, 1));
        if (bytes > 0) {
            memory.setDeviceContents(buffer, buffer.dimensions(), pjrtDevice.id());
        } else {
            buffer.close();
        }
        return MemoryView.of(memory, 0, dt, Layout.rowMajor(shape));
    }
}
