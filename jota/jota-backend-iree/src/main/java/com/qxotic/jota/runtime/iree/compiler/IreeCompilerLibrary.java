package com.qxotic.jota.runtime.iree.compiler;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BOOLEAN;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Consumer;

/**
 * FFM binding of the IREE compiler's stable embedding C API ({@code
 * iree/compiler/embedding_api.h}), loaded from a {@code libIREECompiler} shared library.
 *
 * <p>One instance per library path and process: {@code ireeCompilerGlobalInitialize} is called once
 * when the library is opened and never shut down, as the API requires the global state to outlive
 * every session. The raw handles below mirror the C functions one to one; {@link IreeCompiler} is
 * the safe front end.
 */
public final class IreeCompilerLibrary {

    /** API major version this binding is written against ({@code IREE_COMPILER_API_MAJOR}). */
    public static final int SUPPORTED_API_MAJOR = 1;

    /** {@code IREE_COMPILER_PIPELINE_STD}: the full pipeline, input to VM. */
    static final int PIPELINE_STD = 0;

    /** {@code IREE_COMPILER_DIAGNOSTIC_SEVERITY_*}, by value. */
    static final List<String> SEVERITIES = List.of("note", "warning", "error", "remark");

    private static final Map<Path, IreeCompilerLibrary> OPENED = new ConcurrentHashMap<>();

    /**
     * The JVM's signal handlers, captured when this class initializes, i.e. before any compiler
     * library is loaded; see {@link SignalHandlerGuard}.
     */
    private static final Optional<SignalHandlerGuard> JVM_SIGNAL_HANDLERS =
            SignalHandlerGuard.capture();

    private final Path path;
    private final int apiVersion;
    private final String revision;

    final MethodHandle errorDestroy;
    final MethodHandle errorGetMessage;
    final MethodHandle sessionCreate;
    final MethodHandle sessionDestroy;
    final MethodHandle sessionSetFlags;
    final MethodHandle invocationCreate;
    final MethodHandle invocationDestroy;
    final MethodHandle invocationEnableCallbackDiagnostics;
    final MethodHandle invocationParseSource;
    final MethodHandle invocationSetCompileToPhase;
    final MethodHandle invocationPipeline;
    final MethodHandle invocationOutputIR;
    final MethodHandle invocationOutputVMBytecode;
    final MethodHandle sourceWrapBuffer;
    final MethodHandle sourceDestroy;
    final MethodHandle outputOpenMembuffer;
    final MethodHandle outputMapMemory;
    final MethodHandle outputDestroy;
    private final MethodHandle enumerateHalTargetBackends;
    private final MethodHandle enumeratePlugins;

    private IreeCompilerLibrary(Path path) {
        this.path = path;
        Linker linker = Linker.nativeLinker();
        SymbolLookup lookup = SymbolLookup.libraryLookup(path, Arena.global());
        var b = new Binder(linker, lookup);

        MethodHandle getApiVersion =
                b.bind("ireeCompilerGetAPIVersion", FunctionDescriptor.of(JAVA_INT));
        MethodHandle globalInitialize =
                b.bind("ireeCompilerGlobalInitialize", FunctionDescriptor.ofVoid());
        MethodHandle getRevision =
                b.bind("ireeCompilerGetRevision", FunctionDescriptor.of(ADDRESS));

        errorDestroy = b.bind("ireeCompilerErrorDestroy", FunctionDescriptor.ofVoid(ADDRESS));
        errorGetMessage =
                b.bind("ireeCompilerErrorGetMessage", FunctionDescriptor.of(ADDRESS, ADDRESS));
        sessionCreate = b.bind("ireeCompilerSessionCreate", FunctionDescriptor.of(ADDRESS));
        sessionDestroy = b.bind("ireeCompilerSessionDestroy", FunctionDescriptor.ofVoid(ADDRESS));
        sessionSetFlags =
                b.bind(
                        "ireeCompilerSessionSetFlags",
                        FunctionDescriptor.of(ADDRESS, ADDRESS, JAVA_INT, ADDRESS));
        invocationCreate =
                b.bind("ireeCompilerInvocationCreate", FunctionDescriptor.of(ADDRESS, ADDRESS));
        invocationDestroy =
                b.bind("ireeCompilerInvocationDestroy", FunctionDescriptor.ofVoid(ADDRESS));
        invocationEnableCallbackDiagnostics =
                b.bind(
                        "ireeCompilerInvocationEnableCallbackDiagnostics",
                        FunctionDescriptor.ofVoid(ADDRESS, JAVA_INT, ADDRESS, ADDRESS));
        invocationParseSource =
                b.bind(
                        "ireeCompilerInvocationParseSource",
                        FunctionDescriptor.of(JAVA_BOOLEAN, ADDRESS, ADDRESS));
        invocationSetCompileToPhase =
                b.bind(
                        "ireeCompilerInvocationSetCompileToPhase",
                        FunctionDescriptor.ofVoid(ADDRESS, ADDRESS));
        invocationPipeline =
                b.bind(
                        "ireeCompilerInvocationPipeline",
                        FunctionDescriptor.of(JAVA_BOOLEAN, ADDRESS, JAVA_INT));
        invocationOutputIR =
                b.bind(
                        "ireeCompilerInvocationOutputIR",
                        FunctionDescriptor.of(ADDRESS, ADDRESS, ADDRESS));
        invocationOutputVMBytecode =
                b.bind(
                        "ireeCompilerInvocationOutputVMBytecode",
                        FunctionDescriptor.of(ADDRESS, ADDRESS, ADDRESS));
        sourceWrapBuffer =
                b.bind(
                        "ireeCompilerSourceWrapBuffer",
                        FunctionDescriptor.of(
                                ADDRESS,
                                ADDRESS,
                                ADDRESS,
                                ADDRESS,
                                JAVA_LONG,
                                JAVA_BOOLEAN,
                                ADDRESS));
        sourceDestroy = b.bind("ireeCompilerSourceDestroy", FunctionDescriptor.ofVoid(ADDRESS));
        outputOpenMembuffer =
                b.bind("ireeCompilerOutputOpenMembuffer", FunctionDescriptor.of(ADDRESS, ADDRESS));
        outputMapMemory =
                b.bind(
                        "ireeCompilerOutputMapMemory",
                        FunctionDescriptor.of(ADDRESS, ADDRESS, ADDRESS, ADDRESS));
        outputDestroy = b.bind("ireeCompilerOutputDestroy", FunctionDescriptor.ofVoid(ADDRESS));
        enumerateHalTargetBackends =
                b.bind(
                        "ireeCompilerEnumerateRegisteredHALTargetBackends",
                        FunctionDescriptor.ofVoid(ADDRESS, ADDRESS));
        enumeratePlugins =
                b.bind("ireeCompilerEnumeratePlugins", FunctionDescriptor.ofVoid(ADDRESS, ADDRESS));

        try {
            apiVersion = (int) getApiVersion.invokeExact();
            int major = apiVersion >>> 16;
            if (major != SUPPORTED_API_MAJOR) {
                throw new IreeCompilerException(
                        "Unsupported IREE compiler API version "
                                + major
                                + "."
                                + (apiVersion & 0xFFFF)
                                + " in "
                                + path
                                + " (this binding supports major "
                                + SUPPORTED_API_MAJOR
                                + ")");
            }
            globalInitialize.invokeExact();
            restoreSignalHandlers();
            revision = cString((MemorySegment) getRevision.invokeExact());
        } catch (IreeCompilerException e) {
            throw e;
        } catch (Throwable t) {
            throw new IreeCompilerException("Cannot initialize IREE compiler " + path, t);
        }
    }

    /**
     * Opens (once per process) and initializes the compiler library at {@code path}.
     *
     * @throws IreeCompilerException if the file is missing, not loadable, or has an incompatible
     *     API version
     */
    public static IreeCompilerLibrary open(Path path) {
        Objects.requireNonNull(path, "path");
        if (!Files.isRegularFile(path)) {
            throw new IreeCompilerException("IREE compiler library not found: " + path);
        }
        Path key = path.toAbsolutePath().normalize();
        try {
            return OPENED.computeIfAbsent(key, IreeCompilerLibrary::new);
        } catch (IllegalArgumentException e) {
            // SymbolLookup.libraryLookup: dlopen failure or a missing symbol
            throw new IreeCompilerException("Cannot load IREE compiler library " + key, e);
        }
    }

    /**
     * Puts the JVM's signal handlers back after LLVM replaced them; called after every call into
     * the compiler.
     */
    static void restoreSignalHandlers() {
        JVM_SIGNAL_HANDLERS.ifPresent(SignalHandlerGuard::restore);
    }

    public Path path() {
        return path;
    }

    /** {@code ireeCompilerGetAPIVersion()}: major in the high 16 bits, minor in the low 16. */
    public int apiVersion() {
        return apiVersion;
    }

    public String apiVersionString() {
        return (apiVersion >>> 16) + "." + (apiVersion & 0xFFFF);
    }

    /** {@code ireeCompilerGetRevision()}: the IREE source revision, possibly empty. */
    public String revision() {
        return revision;
    }

    /** HAL target backends compiled into this library, e.g. {@code llvm-cpu}, {@code vmvx}. */
    public List<String> registeredHalTargetBackends() {
        return enumerate(enumerateHalTargetBackends);
    }

    /** Compiler plugins available in this library, e.g. {@code input_stablehlo}. */
    public List<String> registeredPlugins() {
        return enumerate(enumeratePlugins);
    }

    private static List<String> enumerate(MethodHandle function) {
        var names = new ArrayList<String>();
        Consumer<String> sink = names::add;
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment stub =
                    Linker.nativeLinker()
                            .upcallStub(
                                    STRING_CALLBACK.bindTo(sink),
                                    FunctionDescriptor.ofVoid(ADDRESS, ADDRESS),
                                    arena);
            function.invokeExact(stub, MemorySegment.NULL);
            restoreSignalHandlers();
        } catch (Throwable t) {
            throw new IreeCompilerException("IREE compiler enumeration failed", t);
        }
        return List.copyOf(names);
    }

    private static final MethodHandle STRING_CALLBACK;

    static {
        try {
            STRING_CALLBACK =
                    MethodHandles.lookup()
                            .findStatic(
                                    IreeCompilerLibrary.class,
                                    "onString",
                                    MethodType.methodType(
                                            void.class,
                                            Consumer.class,
                                            MemorySegment.class,
                                            MemorySegment.class));
        } catch (ReflectiveOperationException e) {
            throw new ExceptionInInitializerError(e);
        }
    }

    @SuppressWarnings("unused") // upcall target
    private static void onString(Consumer<String> sink, MemorySegment name, MemorySegment user) {
        sink.accept(cString(name));
    }

    static String cString(MemorySegment pointer) {
        if (pointer == null || pointer.equals(MemorySegment.NULL)) {
            return "";
        }
        return pointer.reinterpret(Long.MAX_VALUE).getString(0);
    }

    private record Binder(Linker linker, SymbolLookup lookup) {
        MethodHandle bind(String name, FunctionDescriptor descriptor) {
            MemorySegment symbol =
                    lookup.find(name)
                            .orElseThrow(
                                    () -> new IllegalArgumentException("Missing symbol " + name));
            return linker.downcallHandle(symbol, descriptor);
        }
    }
}
