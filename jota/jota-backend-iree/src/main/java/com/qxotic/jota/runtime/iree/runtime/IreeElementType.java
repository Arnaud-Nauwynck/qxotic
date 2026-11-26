package com.qxotic.jota.runtime.iree.runtime;

import com.qxotic.jota.DataType;

/**
 * {@code iree_hal_element_type_t} values ({@code (numerical_type << 24) | bit_count}) for the jota
 * data types the IREE ABI can carry.
 */
public final class IreeElementType {

    private static final int INTEGER = 0x10;
    private static final int INTEGER_SIGNED = 0x11;
    private static final int INTEGER_UNSIGNED = 0x12;
    private static final int BOOLEAN = 0x13;
    private static final int FLOAT_IEEE = 0x21;
    private static final int FLOAT_BRAIN = 0x22;

    public static final int BOOL_8 = value(BOOLEAN, 8);
    public static final int SINT_8 = value(INTEGER_SIGNED, 8);
    public static final int SINT_16 = value(INTEGER_SIGNED, 16);
    public static final int SINT_32 = value(INTEGER_SIGNED, 32);
    public static final int SINT_64 = value(INTEGER_SIGNED, 64);
    public static final int FLOAT_16 = value(FLOAT_IEEE, 16);
    public static final int BFLOAT_16 = value(FLOAT_BRAIN, 16);
    public static final int FLOAT_32 = value(FLOAT_IEEE, 32);
    public static final int FLOAT_64 = value(FLOAT_IEEE, 64);

    private IreeElementType() {}

    static int value(int numericalType, int bitCount) {
        return numericalType << 24 | bitCount;
    }

    /** The element type IREE expects for {@code type} (jota integers are signed). */
    public static int of(DataType type) {
        if (type == DataType.BOOL) {
            return BOOL_8;
        } else if (type == DataType.I8) {
            return SINT_8;
        } else if (type == DataType.I16) {
            return SINT_16;
        } else if (type == DataType.I32) {
            return SINT_32;
        } else if (type == DataType.I64) {
            return SINT_64;
        } else if (type == DataType.FP16) {
            return FLOAT_16;
        } else if (type == DataType.BF16) {
            return BFLOAT_16;
        } else if (type == DataType.FP32) {
            return FLOAT_32;
        } else if (type == DataType.FP64) {
            return FLOAT_64;
        }
        throw new IllegalArgumentException("No IREE element type for jota data type " + type);
    }

    /** Whether {@link #of(DataType)} maps {@code type}. */
    public static boolean isSupported(DataType type) {
        return type == DataType.BOOL
                || type == DataType.I8
                || type == DataType.I16
                || type == DataType.I32
                || type == DataType.I64
                || type == DataType.FP16
                || type == DataType.BF16
                || type == DataType.FP32
                || type == DataType.FP64;
    }

    /** Bit width of an element type. */
    public static int bitCount(int elementType) {
        return elementType & 0xFF;
    }

    /**
     * Whether IREE's {@code elementType} can be read as jota {@code type}: same width, and the same
     * kind, where IREE's sign-less {@code INT_n} (what some lowerings produce) matches the signed
     * jota integers and an 8-bit integer matches {@code BOOL}.
     */
    public static boolean compatible(int elementType, DataType type) {
        int expected = of(type);
        if (elementType == expected) {
            return true;
        }
        int numerical = elementType >>> 24;
        int bits = bitCount(elementType);
        int expectedNumerical = expected >>> 24;
        if (bits != bitCount(expected)) {
            return false;
        }
        if (expectedNumerical == INTEGER_SIGNED) {
            return numerical == INTEGER;
        }
        if (expectedNumerical == BOOLEAN) {
            return numerical == INTEGER || numerical == INTEGER_SIGNED;
        }
        return false;
    }

    /** E.g. {@code f32}, {@code si32}, {@code 0x21000010}. */
    public static String describe(int elementType) {
        int bits = bitCount(elementType);
        return switch (elementType >>> 24) {
            case INTEGER -> "i" + bits;
            case INTEGER_SIGNED -> "si" + bits;
            case INTEGER_UNSIGNED -> "ui" + bits;
            case BOOLEAN -> "bool" + bits;
            case FLOAT_IEEE -> "f" + bits;
            case FLOAT_BRAIN -> "bf" + bits;
            default -> String.format("0x%08x", elementType);
        };
    }
}
