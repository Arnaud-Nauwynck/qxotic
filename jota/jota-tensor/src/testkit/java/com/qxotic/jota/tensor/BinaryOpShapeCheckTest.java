package com.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jota.Shape;
import com.qxotic.jota.testutil.RunOnAllAvailableBackends;
import com.qxotic.jota.testutil.TensorTestReads;
import org.junit.jupiter.api.Test;

/**
 * Binary ops take operands of equal shapes, or a true scalar: a size-1 dimension is not broadcast
 * implicitly. Shapes of the same rank but different sizes used to pass the check and then read out
 * of bounds, or silently compute a result of the wrong shape.
 */
@RunOnAllAvailableBackends
class BinaryOpShapeCheckTest {

    private static Tensor matrix() {
        return Tensor.of(new float[] {1f, -2f, 0.5f, 0.25f, 3f, -1f}, Shape.of(2, 3));
    }

    private static Tensor column() {
        return Tensor.of(new float[] {10f, 20f}, Shape.of(2, 1));
    }

    @Test
    void sameRankDifferentSizesAreRejected() {
        Tensor a = matrix();
        Tensor col = column();
        Tensor row = Tensor.of(new float[] {10f, 20f, 30f}, Shape.of(1, 3));
        assertThrows(IllegalArgumentException.class, () -> a.subtract(col));
        assertThrows(IllegalArgumentException.class, () -> col.subtract(a));
        assertThrows(IllegalArgumentException.class, () -> a.add(row));
        assertThrows(IllegalArgumentException.class, () -> a.lessThan(col));
    }

    @Test
    void keepDimsReductionIsRejectedWhenTraced() {
        assertThrows(
                IllegalArgumentException.class,
                () -> Tracer.trace(matrix(), x -> x.subtract(x.max(true, 1))));
    }

    @Test
    void explicitExpandBroadcasts() {
        Tensor a = matrix();
        Tensor result = a.subtract(column().expand(Shape.of(2, 3)));
        float[] expected = {-9f, -12f, -9.5f, -19.75f, -17f, -21f};
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], TensorTestReads.readFloat(result, i), "index " + i);
        }
    }
}
