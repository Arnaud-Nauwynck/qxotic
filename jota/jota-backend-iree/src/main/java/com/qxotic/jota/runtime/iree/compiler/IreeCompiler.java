package com.qxotic.jota.runtime.iree.compiler;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import com.qxotic.jota.runtime.iree.IreeConfig;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HexFormat;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Compiles StableHLO (text, MLIR bytecode or a VHLO portable artifact) into an IREE {@code .vmfb}
 * module with the in-process IREE compiler.
 *
 * <p>Each {@link #compile} call uses its own compiler session, so calls are independent and may run
 * on several threads at once. Diagnostics are collected and attached to the exception on failure.
 */
public final class IreeCompiler {

    /** The four bytes MLIR bytecode starts with: {@code ML\xEFR}. */
    private static final byte[] MLIR_BYTECODE_MAGIC = {'M', 'L', (byte) 0xEF, 'R'};

    private static final MethodHandle DIAGNOSTIC_CALLBACK;

    static {
        try {
            DIAGNOSTIC_CALLBACK =
                    MethodHandles.lookup()
                            .findStatic(
                                    IreeCompiler.class,
                                    "onDiagnostic",
                                    MethodType.methodType(
                                            void.class,
                                            List.class,
                                            int.class,
                                            MemorySegment.class,
                                            long.class,
                                            MemorySegment.class));
        } catch (ReflectiveOperationException e) {
            throw new ExceptionInInitializerError(e);
        }
    }

    private final IreeCompilerLibrary library;
    private final List<String> flags;
    private final Optional<Path> dumpDir;

    public IreeCompiler(IreeCompilerLibrary library, List<String> flags, Optional<Path> dumpDir) {
        this.library = Objects.requireNonNull(library, "library");
        this.flags = List.copyOf(flags);
        this.dumpDir = Objects.requireNonNull(dumpDir, "dumpDir");
    }

    /**
     * A compiler for the configured library, target and flags.
     *
     * @throws IreeCompilerException if no compiler library is configured or it cannot be loaded
     */
    public static IreeCompiler fromConfig(IreeConfig config) {
        Path lib =
                config.compilerLibrary()
                        .orElseThrow(
                                () ->
                                        new IreeCompilerException(
                                                "No IREE compiler library configured: "
                                                        + IreeConfig.installHint()));
        return new IreeCompiler(
                IreeCompilerLibrary.open(lib), config.compilerFlags(), config.dumpDir());
    }

    public IreeCompilerLibrary library() {
        return library;
    }

    public List<String> flags() {
        return flags;
    }

    /** Compiles MLIR text (StableHLO / CHLO, or VHLO in its textual form) to a {@code .vmfb}. */
    public CompiledModule compileText(String mlir) {
        return compile(mlir.getBytes(StandardCharsets.UTF_8));
    }

    /**
     * Compiles a source to a {@code .vmfb}: MLIR bytecode (plain StableHLO or a VHLO portable
     * artifact) is recognized by its magic number, anything else is parsed as MLIR text.
     */
    public CompiledModule compile(byte[] source) {
        boolean bytecode = isMlirBytecode(source);
        String name = "jota-" + sha256Prefix(source) + (bytecode ? ".mlirbc" : ".mlir");
        dump(name, source);
        var diagnostics = Collections.synchronizedList(new ArrayList<String>());
        IreeCompilerLibrary l = library;
        // shared: MLIR may report diagnostics from its pass-manager worker threads
        try (Arena arena = Arena.ofShared()) {
            MemorySegment session = MemorySegment.NULL;
            MemorySegment invocation = MemorySegment.NULL;
            MemorySegment src = MemorySegment.NULL;
            MemorySegment output = MemorySegment.NULL;
            try {
                session = (MemorySegment) l.sessionCreate.invokeExact();
                setFlags(session, arena);

                invocation = (MemorySegment) l.invocationCreate.invokeExact(session);
                MemorySegment callback =
                        Linker.nativeLinker()
                                .upcallStub(
                                        DIAGNOSTIC_CALLBACK.bindTo(diagnostics),
                                        FunctionDescriptor.ofVoid(
                                                JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS),
                                        arena);
                l.invocationEnableCallbackDiagnostics.invokeExact(
                        invocation, 0, callback, MemorySegment.NULL);

                // Text must be NUL terminated with the NUL counted in the length; bytecode not.
                MemorySegment buffer = arena.allocate(source.length + (bytecode ? 0 : 1), 16);
                MemorySegment.copy(source, 0, buffer, ValueLayout.JAVA_BYTE, 0, source.length);
                MemorySegment outSource = arena.allocate(ADDRESS);
                check(
                        (MemorySegment)
                                l.sourceWrapBuffer.invokeExact(
                                        session,
                                        arena.allocateFrom(name),
                                        buffer,
                                        buffer.byteSize(),
                                        !bytecode,
                                        outSource),
                        "wrap source",
                        diagnostics);
                src = outSource.get(ADDRESS, 0);

                if (!(boolean) l.invocationParseSource.invokeExact(invocation, src)) {
                    throw new IreeCompilerException("IREE failed to parse " + name, diagnostics);
                }
                if (!(boolean)
                        l.invocationPipeline.invokeExact(
                                invocation, IreeCompilerLibrary.PIPELINE_STD)) {
                    throw new IreeCompilerException("IREE failed to compile " + name, diagnostics);
                }

                MemorySegment outOutput = arena.allocate(ADDRESS);
                check(
                        (MemorySegment) l.outputOpenMembuffer.invokeExact(outOutput),
                        "open output",
                        diagnostics);
                output = outOutput.get(ADDRESS, 0);
                check(
                        (MemorySegment)
                                l.invocationOutputVMBytecode.invokeExact(invocation, output),
                        "write VM bytecode",
                        diagnostics);
                MemorySegment contents = arena.allocate(ADDRESS);
                MemorySegment size = arena.allocate(JAVA_LONG);
                check(
                        (MemorySegment) l.outputMapMemory.invokeExact(output, contents, size),
                        "map output",
                        diagnostics);
                long n = size.get(JAVA_LONG, 0);
                byte[] vmfb =
                        contents.get(ADDRESS, 0).reinterpret(n).toArray(ValueLayout.JAVA_BYTE);
                dump(name + ".vmfb", vmfb);
                return new CompiledModule(vmfb, List.copyOf(diagnostics), flags);
            } finally {
                if (!output.equals(MemorySegment.NULL)) {
                    l.outputDestroy.invokeExact(output);
                }
                if (!src.equals(MemorySegment.NULL)) {
                    l.sourceDestroy.invokeExact(src);
                }
                if (!invocation.equals(MemorySegment.NULL)) {
                    l.invocationDestroy.invokeExact(invocation);
                }
                if (!session.equals(MemorySegment.NULL)) {
                    l.sessionDestroy.invokeExact(session);
                }
                IreeCompilerLibrary.restoreSignalHandlers();
            }
        } catch (IreeCompilerException e) {
            throw e;
        } catch (Throwable t) {
            throw new IreeCompilerException("IREE compiler call failed", diagnostics, t);
        }
    }

    private void setFlags(MemorySegment session, Arena arena) throws Throwable {
        if (flags.isEmpty()) {
            return;
        }
        MemorySegment argv = arena.allocate(ADDRESS, flags.size());
        for (int i = 0; i < flags.size(); i++) {
            argv.setAtIndex(ADDRESS, i, arena.allocateFrom(flags.get(i)));
        }
        check(
                (MemorySegment) library.sessionSetFlags.invokeExact(session, flags.size(), argv),
                "set flags " + flags,
                List.of());
    }

    private void check(MemorySegment error, String what, List<String> diagnostics)
            throws Throwable {
        if (error.equals(MemorySegment.NULL)) {
            return;
        }
        String message =
                IreeCompilerLibrary.cString(
                        (MemorySegment) library.errorGetMessage.invokeExact(error));
        library.errorDestroy.invokeExact(error);
        throw new IreeCompilerException(
                "IREE compiler: cannot " + what + ": " + message, diagnostics);
    }

    @SuppressWarnings("unused") // upcall target
    private static void onDiagnostic(
            List<String> sink,
            int severity,
            MemorySegment message,
            long size,
            MemorySegment userData) {
        String text =
                new String(
                        message.reinterpret(size).toArray(ValueLayout.JAVA_BYTE),
                        StandardCharsets.UTF_8);
        String level =
                severity >= 0 && severity < IreeCompilerLibrary.SEVERITIES.size()
                        ? IreeCompilerLibrary.SEVERITIES.get(severity)
                        : "severity " + severity;
        sink.add(level + ": " + text);
    }

    static boolean isMlirBytecode(byte[] source) {
        if (source.length < MLIR_BYTECODE_MAGIC.length) {
            return false;
        }
        for (int i = 0; i < MLIR_BYTECODE_MAGIC.length; i++) {
            if (source[i] != MLIR_BYTECODE_MAGIC[i]) {
                return false;
            }
        }
        return true;
    }

    private void dump(String name, byte[] bytes) {
        if (dumpDir.isEmpty()) {
            return;
        }
        try {
            Files.createDirectories(dumpDir.get());
            Files.write(dumpDir.get().resolve(name), bytes);
        } catch (IOException e) {
            throw new UncheckedIOException("Cannot dump " + name + " to " + dumpDir.get(), e);
        }
    }

    private static String sha256Prefix(byte[] bytes) {
        try {
            byte[] digest = MessageDigest.getInstance("SHA-256").digest(bytes);
            return HexFormat.of().formatHex(digest, 0, 8);
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException(e);
        }
    }

    /**
     * A compiled IREE VM module ({@code .vmfb} bytes), with the flags and diagnostics of its
     * compile.
     */
    public record CompiledModule(byte[] vmfb, List<String> diagnostics, List<String> flags) {

        /**
         * Whether the bytes look like a VM module: either a bare FlatBuffer with the {@code IREE}
         * file identifier at offset 4, or the usual {@code .vmfb} archive, a ZIP (stored entries,
         * so the runtime can map it in place) whose first entry is {@code module.fb}.
         */
        public boolean hasVmfbIdentifier() {
            if (vmfb.length >= 8 && matches(vmfb, 4, "IREE")) {
                return true;
            }
            if (vmfb.length < 30 || !matches(vmfb, 0, "PK\u0003\u0004")) {
                return false;
            }
            int nameLength = (vmfb[26] & 0xFF) | (vmfb[27] & 0xFF) << 8;
            return nameLength == "module.fb".length()
                    && vmfb.length >= 30 + nameLength
                    && matches(vmfb, 30, "module.fb");
        }

        private static boolean matches(byte[] bytes, int offset, String ascii) {
            for (int i = 0; i < ascii.length(); i++) {
                if (bytes[offset + i] != (byte) ascii.charAt(i)) {
                    return false;
                }
            }
            return true;
        }
    }
}
