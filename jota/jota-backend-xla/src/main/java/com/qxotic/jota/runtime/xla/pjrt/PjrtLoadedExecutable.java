package com.qxotic.jota.runtime.xla.pjrt;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;

/** A {@code PJRT_LoadedExecutable}: a program compiled for, and loaded on, one device. */
public final class PjrtLoadedExecutable implements AutoCloseable {

    /** PJRT_ExecuteOptions of header 0.81 (14 fields), all defaults. */
    private static final int EXECUTE_OPTIONS_SLOTS = 14;

    private final PjrtApi api;
    private MemorySegment handle;
    private final int numOutputs;

    PjrtLoadedExecutable(PjrtApi api, MemorySegment handle) {
        this.api = api;
        this.handle = handle;
        this.numOutputs = queryNumOutputs();
    }

    /** The number of results of {@code @main} (tuples are untupled by XLA). */
    public int numOutputs() {
        return numOutputs;
    }

    /**
     * Runs the executable on its device with {@code arguments} (in {@code @main} parameter order)
     * and waits for completion. The caller owns, and must close, the returned buffers.
     */
    public List<PjrtBuffer> execute(List<PjrtBuffer> arguments) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment options = Structs.alloc(arena, EXECUTE_OPTIONS_SLOTS);

            MemorySegment argList = arena.allocate(ADDRESS, Math.max(1, arguments.size()));
            for (int i = 0; i < arguments.size(); i++) {
                argList.setAtIndex(ADDRESS, i, arguments.get(i).handle());
            }
            MemorySegment argLists = arena.allocate(ADDRESS, 1);
            argLists.setAtIndex(ADDRESS, 0, argList);

            MemorySegment outList = arena.allocate(ADDRESS, Math.max(1, numOutputs));
            MemorySegment outLists = arena.allocate(ADDRESS, 1);
            outLists.setAtIndex(ADDRESS, 0, outList);
            MemorySegment events = arena.allocate(ADDRESS, 1);

            // PJRT_LoadedExecutable_Execute_Args: size, ext, executable, options, argument_lists,
            // num_devices, num_args, output_lists, device_complete_events, execute_device
            MemorySegment args = Structs.alloc(arena, 10);
            Structs.ptr(args, 2, handle());
            Structs.ptr(args, 3, options);
            Structs.ptr(args, 4, argLists);
            Structs.u64(args, 5, 1);
            Structs.u64(args, 6, arguments.size());
            Structs.ptr(args, 7, outLists);
            Structs.ptr(args, 8, events);
            api.check("PJRT_LoadedExecutable_Execute", args);

            List<PjrtBuffer> outputs = new ArrayList<>(numOutputs);
            for (int i = 0; i < numOutputs; i++) {
                outputs.add(new PjrtBuffer(api, outList.getAtIndex(ADDRESS, i)));
            }
            try {
                api.awaitAndDestroy("PJRT_LoadedExecutable_Execute", events.getAtIndex(ADDRESS, 0));
            } catch (RuntimeException e) {
                outputs.forEach(PjrtBuffer::close);
                throw e;
            }
            return outputs;
        }
    }

    private int queryNumOutputs() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment get = Structs.alloc(arena, 4);
            Structs.ptr(get, 2, handle());
            api.check("PJRT_LoadedExecutable_GetExecutable", get);
            MemorySegment executable = Structs.ptr(get, 3);
            try {
                MemorySegment num = Structs.alloc(arena, 4);
                Structs.ptr(num, 2, executable);
                api.check("PJRT_Executable_NumOutputs", num);
                return Math.toIntExact(Structs.u64(num, 3));
            } finally {
                api.destroy("PJRT_Executable_Destroy", executable);
            }
        }
    }

    /**
     * {@code PJRT_Executable_Serialize}: bytes that {@link PjrtClient#deserialize} turns back into
     * an executable, with the same plugin build (the format is plugin-specific).
     */
    public byte[] serialize() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment get = Structs.alloc(arena, 4);
            Structs.ptr(get, 2, handle());
            api.check("PJRT_LoadedExecutable_GetExecutable", get);
            MemorySegment executable = Structs.ptr(get, 3);
            try {
                // PJRT_Executable_Serialize_Args: size, ext, executable, serialized_bytes,
                // serialized_bytes_size, serialized_executable, serialized_executable_deleter
                MemorySegment args = Structs.alloc(arena, 7);
                Structs.ptr(args, 2, executable);
                api.check("PJRT_Executable_Serialize", args);
                MemorySegment backing = Structs.ptr(args, 5);
                try {
                    long size = Structs.u64(args, 4);
                    return Structs.ptr(args, 3).reinterpret(size).toArray(JAVA_BYTE);
                } finally {
                    api.callPointer(Structs.ptr(args, 6), backing);
                }
            } finally {
                api.destroy("PJRT_Executable_Destroy", executable);
            }
        }
    }

    private MemorySegment handle() {
        if (handle == null) {
            throw new IllegalStateException("executable is closed");
        }
        return handle;
    }

    @Override
    public void close() {
        if (handle != null) {
            MemorySegment h = handle;
            handle = null;
            api.destroy("PJRT_LoadedExecutable_Destroy", h);
        }
    }
}
