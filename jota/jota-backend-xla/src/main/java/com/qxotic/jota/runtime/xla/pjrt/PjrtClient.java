package com.qxotic.jota.runtime.xla.pjrt;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/** A {@code PJRT_Client}: the devices of one plugin, compilation and host transfers. */
public final class PjrtClient implements AutoCloseable {

    /** {@code PJRT_HostBufferSemantics_kImmutableOnlyDuringCall}. */
    private static final int IMMUTABLE_ONLY_DURING_CALL = 0;

    private final PjrtApi api;
    private MemorySegment handle;
    private final List<PjrtDevice> devices;

    private PjrtClient(PjrtApi api, MemorySegment handle) {
        this.api = api;
        this.handle = handle;
        this.devices = queryDevices();
    }

    /** Initializes the plugin if needed and creates a client with default options. */
    public static PjrtClient create(PjrtApi api) {
        api.initialize();
        try (Arena arena = Arena.ofConfined()) {
            // PJRT_Client_Create_Args: size, ext, create_options, num_options, kv_get, kv_get_arg,
            // kv_put, kv_put_arg, client (out), kv_try_get, kv_try_get_arg
            MemorySegment args = Structs.alloc(arena, 11);
            api.check("PJRT_Client_Create", args);
            return new PjrtClient(api, Structs.ptr(args, 8));
        }
    }

    public PjrtApi api() {
        return api;
    }

    /** {@code PJRT_Client_PlatformName}, e.g. {@code "cpu"}, {@code "cuda"}. */
    public String platformName() {
        return stringQuery("PJRT_Client_PlatformName");
    }

    /** {@code PJRT_Client_PlatformVersion}. */
    public String platformVersion() {
        return stringQuery("PJRT_Client_PlatformVersion");
    }

    /** The addressable devices. */
    public List<PjrtDevice> devices() {
        return devices;
    }

    /**
     * {@code PJRT_Executable_DeserializeAndLoad}: loads bytes from {@link
     * PjrtLoadedExecutable#serialize}. {@code compileOptions} overrides the options recorded in the
     * executable (null keeps them).
     */
    public PjrtLoadedExecutable deserialize(byte[] serialized, byte[] compileOptions) {
        try (Arena arena = Arena.ofConfined()) {
            // PJRT_Executable_DeserializeAndLoad_Args: size, ext, client, serialized_executable,
            // serialized_executable_size, loaded_executable (out),
            // overridden_serialized_compile_options, overridden_serialized_compile_options_size
            MemorySegment args = Structs.alloc(arena, 8);
            Structs.ptr(args, 2, handle());
            Structs.ptr(args, 3, Structs.bytes(arena, serialized));
            Structs.u64(args, 4, serialized.length);
            if (compileOptions != null) {
                Structs.ptr(args, 6, Structs.bytes(arena, compileOptions));
                Structs.u64(args, 7, compileOptions.length);
            }
            api.check("PJRT_Executable_DeserializeAndLoad", args);
            return new PjrtLoadedExecutable(api, Structs.ptr(args, 5));
        }
    }

    /**
     * Compiles {@code code}. {@code format} is {@code "mlir"} (StableHLO text, plain bytecode or
     * portable artifact) or {@code "hlo"}; {@code compileOptions} is a serialized {@code
     * CompileOptionsProto} (see {@code CompileOptions}), which also picks the device.
     */
    public PjrtLoadedExecutable compile(byte[] code, String format, byte[] compileOptions) {
        try (Arena arena = Arena.ofConfined()) {
            byte[] fmt = format.getBytes(StandardCharsets.UTF_8);
            // PJRT_Program: size, ext, code, code_size, format, format_size
            MemorySegment program = Structs.alloc(arena, 6);
            Structs.ptr(program, 2, Structs.bytes(arena, code));
            Structs.u64(program, 3, code.length);
            Structs.ptr(program, 4, Structs.bytes(arena, fmt));
            Structs.u64(program, 5, fmt.length);
            // PJRT_Client_Compile_Args: size, ext, client, program, compile_options,
            // compile_options_size, executable (out)
            MemorySegment args = Structs.alloc(arena, 7);
            Structs.ptr(args, 2, handle());
            Structs.ptr(args, 3, program);
            Structs.ptr(args, 4, Structs.bytes(arena, compileOptions));
            Structs.u64(args, 5, compileOptions.length);
            api.check("PJRT_Client_Compile", args);
            return new PjrtLoadedExecutable(api, Structs.ptr(args, 6));
        }
    }

    /**
     * Copies a dense row-major host array to {@code device}; {@code data} may be released when this
     * returns.
     */
    public PjrtBuffer bufferFromHost(
            MemorySegment data, PjrtBufferType type, long[] dims, PjrtDevice device) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment dimSeg = arena.allocate(JAVA_LONG, Math.max(1, dims.length));
            for (int i = 0; i < dims.length; i++) {
                dimSeg.setAtIndex(JAVA_LONG, i, dims[i]);
            }
            // PJRT_Client_BufferFromHostBuffer_Args: size, ext, client, data, type, dims,
            // num_dims, byte_strides, num_byte_strides, host_buffer_semantics, device, memory,
            // device_layout, done_with_host_buffer (out), buffer (out)
            MemorySegment args = Structs.alloc(arena, 15);
            Structs.ptr(args, 2, handle());
            Structs.ptr(args, 3, data);
            Structs.i32(args, 4, type.value());
            Structs.ptr(args, 5, dimSeg);
            Structs.u64(args, 6, dims.length);
            Structs.i32(args, 9, IMMUTABLE_ONLY_DURING_CALL);
            Structs.ptr(args, 10, device.handle());
            api.check("PJRT_Client_BufferFromHostBuffer", args);
            PjrtBuffer buffer = new PjrtBuffer(api, Structs.ptr(args, 14));
            try {
                api.awaitAndDestroy("PJRT_Client_BufferFromHostBuffer", Structs.ptr(args, 13));
            } catch (RuntimeException e) {
                buffer.close();
                throw e;
            }
            return buffer;
        }
    }

    private List<PjrtDevice> queryDevices() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment args = Structs.alloc(arena, 5);
            Structs.ptr(args, 2, handle());
            api.check("PJRT_Client_AddressableDevices", args);
            long n = Structs.u64(args, 4);
            MemorySegment array = Structs.ptr(args, 3).reinterpret(n * ADDRESS.byteSize());
            List<PjrtDevice> list = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                MemorySegment device = array.getAtIndex(ADDRESS, i);
                MemorySegment description = deviceDescription(arena, device);
                list.add(
                        new PjrtDevice(
                                device,
                                i,
                                deviceId(arena, description),
                                deviceKind(arena, description)));
            }
            return List.copyOf(list);
        }
    }

    private MemorySegment deviceDescription(Arena arena, MemorySegment device) {
        MemorySegment get = Structs.alloc(arena, 4);
        Structs.ptr(get, 2, device);
        api.check("PJRT_Device_GetDescription", get);
        return Structs.ptr(get, 3);
    }

    private int deviceId(Arena arena, MemorySegment description) {
        // PJRT_DeviceDescription_Id_Args: size, ext, device_description, id (int, last)
        MemorySegment id = Structs.allocEndingInInt(arena, 4);
        Structs.ptr(id, 2, description);
        api.check("PJRT_DeviceDescription_Id", id);
        return Structs.i32(id, 3);
    }

    private String deviceKind(Arena arena, MemorySegment description) {
        MemorySegment kind = Structs.alloc(arena, 5);
        Structs.ptr(kind, 2, description);
        api.check("PJRT_DeviceDescription_Kind", kind);
        return Structs.string(Structs.ptr(kind, 3), Structs.u64(kind, 4));
    }

    private String stringQuery(String function) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment args = Structs.alloc(arena, 5);
            Structs.ptr(args, 2, handle());
            api.check(function, args);
            return Structs.string(Structs.ptr(args, 3), Structs.u64(args, 4));
        }
    }

    private MemorySegment handle() {
        if (handle == null) {
            throw new IllegalStateException("client is closed");
        }
        return handle;
    }

    @Override
    public void close() {
        if (handle != null) {
            MemorySegment h = handle;
            handle = null;
            api.destroy("PJRT_Client_Destroy", h);
        }
    }
}
