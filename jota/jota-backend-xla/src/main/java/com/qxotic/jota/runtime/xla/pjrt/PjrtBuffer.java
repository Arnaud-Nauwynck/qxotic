package com.qxotic.jota.runtime.xla.pjrt;

import static java.lang.foreign.ValueLayout.JAVA_LONG;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

/** A {@code PJRT_Buffer}: an array on a device, dense, in XLA's default (row-major) layout. */
public final class PjrtBuffer implements AutoCloseable {

    private final PjrtApi api;
    private MemorySegment handle;

    PjrtBuffer(PjrtApi api, MemorySegment handle) {
        this.api = api;
        this.handle = handle;
    }

    MemorySegment handle() {
        if (handle == null) {
            throw new IllegalStateException("buffer is closed");
        }
        return handle;
    }

    /** The element type. */
    public PjrtBufferType type() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment args = Structs.allocEndingInInt(arena, 4);
            Structs.ptr(args, 2, handle());
            api.check("PJRT_Buffer_ElementType", args);
            return PjrtBufferType.of(Structs.i32(args, 3));
        }
    }

    /** The dimensions (empty for a scalar). */
    public long[] dimensions() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment args = Structs.alloc(arena, 5);
            Structs.ptr(args, 2, handle());
            api.check("PJRT_Buffer_Dimensions", args);
            long n = Structs.u64(args, 4);
            return n == 0
                    ? new long[0]
                    : Structs.ptr(args, 3).reinterpret(n * 8).toArray(JAVA_LONG);
        }
    }

    /** The size in bytes of the host copy {@link #copyToHost} writes. */
    public long hostSize() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment args = toHostArgs(arena, MemorySegment.NULL, 0);
            api.check("PJRT_Buffer_ToHostBuffer", args);
            return Structs.u64(args, 5);
        }
    }

    /** Copies the buffer into {@code dst} (row-major, dense) and waits for the copy. */
    public void copyToHost(MemorySegment dst) {
        long size = hostSize();
        if (dst.byteSize() < size) {
            throw new IllegalArgumentException(
                    "destination has " + dst.byteSize() + " bytes, buffer needs " + size);
        }
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment args = toHostArgs(arena, dst, size);
            api.check("PJRT_Buffer_ToHostBuffer", args);
            api.awaitAndDestroy("PJRT_Buffer_ToHostBuffer", Structs.ptr(args, 6));
        }
    }

    /** The buffer's contents in a new segment of {@code arena}. */
    public MemorySegment toHost(Arena arena) {
        MemorySegment dst = arena.allocate(Math.max(1, hostSize()), 64);
        copyToHost(dst);
        return dst.asSlice(0, hostSize());
    }

    // PJRT_Buffer_ToHostBuffer_Args: size, ext, src, host_layout, dst, dst_size, event
    private MemorySegment toHostArgs(Arena arena, MemorySegment dst, long size) {
        MemorySegment args = Structs.alloc(arena, 7);
        Structs.ptr(args, 2, handle());
        Structs.ptr(args, 4, dst);
        Structs.u64(args, 5, size);
        return args;
    }

    @Override
    public void close() {
        if (handle != null) {
            MemorySegment h = handle;
            handle = null;
            api.destroy("PJRT_Buffer_Destroy", h);
        }
    }
}
