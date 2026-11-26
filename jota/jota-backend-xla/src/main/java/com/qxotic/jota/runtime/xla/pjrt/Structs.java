package com.qxotic.jota.runtime.xla.pjrt;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.charset.StandardCharsets;

/**
 * Access to PJRT argument structs by 8-byte slot.
 *
 * <p>On the 64-bit ABIs PJRT supports, every field of the argument structs this package uses is a
 * pointer, a {@code size_t}, an {@code int64_t}, or a 4-byte enum / {@code int} followed by a
 * pointer-aligned field (so padded to 8 bytes): field {@code k} lives at offset {@code 8 * k}. Slot
 * 0 is {@code struct_size}, slot 1 {@code extension_start}.
 *
 * <p>{@code struct_size} is the {@code PJRT_STRUCT_SIZE} of the header the layout follows (offset
 * plus size of the last field), which tells the plugin which fields the caller knows.
 */
final class Structs {

    static final long SLOT = 8;

    private Structs() {}

    /** A zeroed struct of {@code slots} fields whose last field is pointer sized. */
    static MemorySegment alloc(Arena arena, int slots) {
        return alloc(arena, slots, slots * SLOT);
    }

    /** A zeroed struct of {@code slots} fields whose last field is a 4-byte enum or int. */
    static MemorySegment allocEndingInInt(Arena arena, int slots) {
        return alloc(arena, slots, (slots - 1) * SLOT + 4);
    }

    private static MemorySegment alloc(Arena arena, int slots, long structSize) {
        MemorySegment s = arena.allocate(slots * SLOT, SLOT);
        s.set(JAVA_LONG, 0, structSize);
        return s;
    }

    static void ptr(MemorySegment s, int slot, MemorySegment value) {
        s.set(ADDRESS, slot * SLOT, value);
    }

    static MemorySegment ptr(MemorySegment s, int slot) {
        return s.get(ADDRESS, slot * SLOT);
    }

    static void u64(MemorySegment s, int slot, long value) {
        s.set(JAVA_LONG, slot * SLOT, value);
    }

    static long u64(MemorySegment s, int slot) {
        return s.get(JAVA_LONG, slot * SLOT);
    }

    static void i32(MemorySegment s, int slot, int value) {
        s.set(JAVA_INT, slot * SLOT, value);
    }

    static int i32(MemorySegment s, int slot) {
        return s.get(JAVA_INT, slot * SLOT);
    }

    /** A UTF-8 string given as pointer and byte length (not NUL-terminated). */
    static String string(MemorySegment ptr, long size) {
        if (ptr.equals(MemorySegment.NULL) || size == 0) {
            return "";
        }
        return new String(ptr.reinterpret(size).toArray(JAVA_BYTE), StandardCharsets.UTF_8);
    }

    /** {@code bytes} copied into {@code arena} (at least one byte, so never NULL). */
    static MemorySegment bytes(Arena arena, byte[] bytes) {
        MemorySegment s = arena.allocate(Math.max(1, bytes.length));
        MemorySegment.copy(bytes, 0, s, JAVA_BYTE, 0, bytes.length);
        return s;
    }
}
