package com.qxotic.jota.runtime.iree.runtime;

import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.Objects;

/**
 * A dense row-major tensor in host memory, as passed to and returned by IREE calls.
 *
 * @param elementType an {@code iree_hal_element_type_t} ({@link IreeElementType})
 * @param shape dimensions, outermost first; empty for a scalar
 * @param data exactly {@code product(shape) * bitCount / 8} bytes
 */
public record HostTensor(int elementType, long[] shape, MemorySegment data) {

    public HostTensor {
        Objects.requireNonNull(shape, "shape");
        Objects.requireNonNull(data, "data");
        shape = shape.clone();
        long elements = 1;
        for (long d : shape) {
            if (d < 0) {
                throw new IllegalArgumentException(
                        "negative dimension in " + Arrays.toString(shape));
            }
            elements = Math.multiplyExact(elements, d);
        }
        long expected = Math.ceilDiv(elements * IreeElementType.bitCount(elementType), 8);
        if (data.byteSize() != expected) {
            throw new IllegalArgumentException(
                    "tensor "
                            + Arrays.toString(shape)
                            + " of "
                            + IreeElementType.describe(elementType)
                            + " needs "
                            + expected
                            + " bytes, got "
                            + data.byteSize());
        }
    }

    @Override
    public long[] shape() {
        return shape.clone();
    }

    @Override
    public String toString() {
        return "HostTensor["
                + IreeElementType.describe(elementType)
                + Arrays.toString(shape)
                + ", "
                + data.byteSize()
                + " bytes]";
    }
}
