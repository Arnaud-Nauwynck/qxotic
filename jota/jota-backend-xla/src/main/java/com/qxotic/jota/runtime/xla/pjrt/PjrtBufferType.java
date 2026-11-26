package com.qxotic.jota.runtime.xla.pjrt;

/** {@code PJRT_Buffer_Type}, in declaration order (the ordinal is the C value). */
public enum PjrtBufferType {
    INVALID(0),
    PRED(8),
    S8(8),
    S16(16),
    S32(32),
    S64(64),
    U8(8),
    U16(16),
    U32(32),
    U64(64),
    F16(16),
    F32(32),
    F64(64),
    BF16(16),
    C64(64),
    C128(128),
    F8E5M2(8),
    F8E4M3FN(8),
    F8E4M3B11FNUZ(8),
    F8E5M2FNUZ(8),
    F8E4M3FNUZ(8),
    S4(4),
    U4(4),
    TOKEN(0),
    S2(2),
    U2(2),
    F8E4M3(8),
    F8E3M4(8),
    F8E8M0FNU(8),
    F4E2M1FN(4);

    private final int bits;

    PjrtBufferType(int bits) {
        this.bits = bits;
    }

    /** Storage bits of one element ({@code PRED} is one byte; 0 for INVALID and TOKEN). */
    public int bits() {
        return bits;
    }

    /** The C enum value. */
    public int value() {
        return ordinal();
    }

    /** The type of C enum value {@code value}. */
    public static PjrtBufferType of(int value) {
        PjrtBufferType[] all = values();
        if (value < 0 || value >= all.length) {
            throw new IllegalArgumentException("unknown PJRT_Buffer_Type " + value);
        }
        return all[value];
    }
}
