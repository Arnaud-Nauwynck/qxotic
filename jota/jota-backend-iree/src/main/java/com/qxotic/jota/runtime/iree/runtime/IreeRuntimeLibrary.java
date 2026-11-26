package com.qxotic.jota.runtime.iree.runtime;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;

/**
 * FFM binding of {@code libjota_iree_rt}: the IREE C runtime built as one shared library by {@code
 * src/main/native}, plus the {@code jota_iree_rt_*} helpers declared in {@code jota_iree_rt.c}. One
 * instance per library path and process.
 */
public final class IreeRuntimeLibrary {

    /** {@code jota_iree_rt_abi_version()} this binding expects. */
    public static final int SHIM_ABI_VERSION = 1;

    private static final Map<Path, IreeRuntimeLibrary> OPENED = new ConcurrentHashMap<>();

    private final Path path;

    final MethodHandle statusConsume;
    final MethodHandle statusCode;
    final MethodHandle free;
    final MethodHandle instanceCreate;
    final MethodHandle instanceRelease;
    final MethodHandle deviceCreate;
    final MethodHandle deviceRelease;
    final MethodHandle sessionCreate;
    final MethodHandle sessionRelease;
    final MethodHandle sessionAppendModule;
    final MethodHandle callCreate;
    final MethodHandle callRelease;
    final MethodHandle callPushInput;
    final MethodHandle callInvoke;
    final MethodHandle callOutputCount;
    final MethodHandle callPopOutput;
    final MethodHandle bufferViewRank;
    final MethodHandle bufferViewDim;
    final MethodHandle bufferViewElementType;
    final MethodHandle bufferViewByteLength;
    final MethodHandle bufferViewRead;
    final MethodHandle bufferViewRelease;

    private IreeRuntimeLibrary(Path path) {
        this.path = path;
        Linker linker = Linker.nativeLinker();
        SymbolLookup lookup = SymbolLookup.libraryLookup(path, Arena.global());
        java.util.function.BiFunction<String, FunctionDescriptor, MethodHandle> bind =
                (name, fd) ->
                        linker.downcallHandle(
                                lookup.find(name)
                                        .orElseThrow(
                                                () ->
                                                        new IllegalArgumentException(
                                                                "Missing symbol " + name)),
                                fd);

        MethodHandle abiVersion =
                bind.apply("jota_iree_rt_abi_version", FunctionDescriptor.of(JAVA_INT));
        statusConsume =
                bind.apply("jota_iree_rt_status_consume", FunctionDescriptor.of(ADDRESS, ADDRESS));
        statusCode =
                bind.apply("jota_iree_rt_status_code", FunctionDescriptor.of(JAVA_INT, ADDRESS));
        free = bind.apply("jota_iree_rt_free", FunctionDescriptor.ofVoid(ADDRESS));
        instanceCreate =
                bind.apply("jota_iree_rt_instance_create", FunctionDescriptor.of(ADDRESS, ADDRESS));
        instanceRelease =
                bind.apply("iree_runtime_instance_release", FunctionDescriptor.ofVoid(ADDRESS));
        deviceCreate =
                bind.apply(
                        "jota_iree_rt_device_create",
                        FunctionDescriptor.of(ADDRESS, ADDRESS, ADDRESS, ADDRESS));
        deviceRelease = bind.apply("iree_hal_device_release", FunctionDescriptor.ofVoid(ADDRESS));
        sessionCreate =
                bind.apply(
                        "jota_iree_rt_session_create",
                        FunctionDescriptor.of(ADDRESS, ADDRESS, ADDRESS, ADDRESS));
        sessionRelease =
                bind.apply("iree_runtime_session_release", FunctionDescriptor.ofVoid(ADDRESS));
        sessionAppendModule =
                bind.apply(
                        "jota_iree_rt_session_append_module",
                        FunctionDescriptor.of(ADDRESS, ADDRESS, ADDRESS, JAVA_LONG));
        callCreate =
                bind.apply(
                        "jota_iree_rt_call_create",
                        FunctionDescriptor.of(ADDRESS, ADDRESS, ADDRESS, ADDRESS));
        callRelease = bind.apply("jota_iree_rt_call_release", FunctionDescriptor.ofVoid(ADDRESS));
        callPushInput =
                bind.apply(
                        "jota_iree_rt_call_push_input",
                        FunctionDescriptor.of(
                                ADDRESS, ADDRESS, ADDRESS, JAVA_LONG, JAVA_INT, ADDRESS,
                                JAVA_LONG));
        callInvoke =
                bind.apply("jota_iree_rt_call_invoke", FunctionDescriptor.of(ADDRESS, ADDRESS));
        callOutputCount =
                bind.apply(
                        "jota_iree_rt_call_output_count",
                        FunctionDescriptor.of(JAVA_LONG, ADDRESS));
        callPopOutput =
                bind.apply(
                        "jota_iree_rt_call_pop_output",
                        FunctionDescriptor.of(ADDRESS, ADDRESS, ADDRESS));
        bufferViewRank =
                bind.apply(
                        "jota_iree_rt_buffer_view_rank", FunctionDescriptor.of(JAVA_LONG, ADDRESS));
        bufferViewDim =
                bind.apply(
                        "jota_iree_rt_buffer_view_dim",
                        FunctionDescriptor.of(JAVA_LONG, ADDRESS, JAVA_LONG));
        bufferViewElementType =
                bind.apply(
                        "jota_iree_rt_buffer_view_element_type",
                        FunctionDescriptor.of(JAVA_INT, ADDRESS));
        bufferViewByteLength =
                bind.apply(
                        "jota_iree_rt_buffer_view_byte_length",
                        FunctionDescriptor.of(JAVA_LONG, ADDRESS));
        bufferViewRead =
                bind.apply(
                        "jota_iree_rt_buffer_view_read",
                        FunctionDescriptor.of(ADDRESS, ADDRESS, ADDRESS, ADDRESS, JAVA_LONG));
        bufferViewRelease =
                bind.apply("iree_hal_buffer_view_release", FunctionDescriptor.ofVoid(ADDRESS));

        int abi;
        try {
            abi = (int) abiVersion.invokeExact();
        } catch (Throwable t) {
            throw new IreeRuntimeException("Cannot query " + path, t);
        }
        if (abi != SHIM_ABI_VERSION) {
            throw new IreeRuntimeException(
                    path
                            + " has shim ABI "
                            + abi
                            + ", this binding expects "
                            + SHIM_ABI_VERSION
                            + ": rebuild src/main/native");
        }
    }

    /**
     * Opens (once per process) the runtime library at {@code path}.
     *
     * @throws IreeRuntimeException if it is missing, not loadable or from another shim version
     */
    public static IreeRuntimeLibrary open(Path path) {
        Objects.requireNonNull(path, "path");
        if (!Files.isRegularFile(path)) {
            throw new IreeRuntimeException("IREE runtime library not found: " + path);
        }
        Path key = path.toAbsolutePath().normalize();
        try {
            return OPENED.computeIfAbsent(key, IreeRuntimeLibrary::new);
        } catch (IllegalArgumentException e) {
            throw new IreeRuntimeException("Cannot load IREE runtime library " + key, e);
        }
    }

    public Path path() {
        return path;
    }

    /** Throws for a non-OK {@code iree_status_t} (a non-null pointer), consuming it. */
    void check(MemorySegment status, String what) {
        if (status.equals(MemorySegment.NULL)) {
            return;
        }
        try {
            int code = (int) statusCode.invokeExact(status);
            MemorySegment text = (MemorySegment) statusConsume.invokeExact(status);
            String message = "";
            if (!text.equals(MemorySegment.NULL)) {
                message = text.reinterpret(Long.MAX_VALUE).getString(0);
                free.invokeExact(text);
            }
            throw new IreeRuntimeException(what + " failed: " + message, code);
        } catch (IreeRuntimeException e) {
            throw e;
        } catch (Throwable t) {
            throw new IreeRuntimeException(what + " failed", t);
        }
    }
}
