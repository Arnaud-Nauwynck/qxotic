package com.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import com.qxotic.jota.testutil.RunOnAllAvailableBackends;
import com.qxotic.jota.testutil.TensorTestReads;
import java.util.List;
import org.junit.jupiter.api.Test;

@RunOnAllAvailableBackends
class ComparisonOpTest {

    private static final List<DataType> PRIMITIVE_DATA_TYPES =
            List.of(
                    DataType.BOOL,
                    DataType.I8,
                    DataType.I16,
                    DataType.I32,
                    DataType.I64,
                    DataType.FP16,
                    DataType.BF16,
                    DataType.FP32,
                    DataType.FP64);

    @Test
    void derivedComparisonsWorkAcrossTypes() {
        Shape shape = Shape.of(4);
        for (DataType dataType : PRIMITIVE_DATA_TYPES) {
            Tensor leftTensor = range(dataType, shape);
            Tensor rightTensor = fullZeros(dataType, shape);

            Tensor notEqual = Tracer.trace(leftTensor, rightTensor, Tensor::notEqual);
            Tensor greaterThan = Tracer.trace(leftTensor, rightTensor, Tensor::greaterThan);
            Tensor lessThanOrEqual = Tracer.trace(leftTensor, rightTensor, Tensor::lessThanOrEqual);
            Tensor greaterThanOrEqual =
                    Tracer.trace(leftTensor, rightTensor, Tensor::greaterThanOrEqual);

            for (int i = 0; i < shape.size(); i++) {
                int leftValue = dataType == DataType.BOOL ? (i % 2) : i;
                int rightValue = 0;
                assertEquals(leftValue != rightValue ? 1 : 0, readBool(notEqual, i));
                assertEquals(leftValue > rightValue ? 1 : 0, readBool(greaterThan, i));
                assertEquals(leftValue <= rightValue ? 1 : 0, readBool(lessThanOrEqual, i));
                assertEquals(leftValue >= rightValue ? 1 : 0, readBool(greaterThanOrEqual, i));
            }

            Tensor flippedGreaterThan = Tracer.trace(rightTensor, leftTensor, Tensor::greaterThan);
            for (int i = 0; i < shape.size(); i++) {
                int leftValue = dataType == DataType.BOOL ? (i % 2) : i;
                int rightValue = 0;
                assertEquals(rightValue > leftValue ? 1 : 0, readBool(flippedGreaterThan, i));
            }
        }
    }

    /** Every ordered comparison with a NaN operand is false, and only != is true (IEEE 754). */
    @Test
    void comparisonsWithNaNFollowIeee() {
        Shape shape = Shape.of(4);
        for (DataType dataType :
                List.of(DataType.FP16, DataType.BF16, DataType.FP32, DataType.FP64)) {
            Tensor left =
                    Tensor.of(new float[] {Float.NaN, 1f, Float.NaN, 2f}, shape).cast(dataType);
            Tensor right =
                    Tensor.of(new float[] {1f, Float.NaN, Float.NaN, 2f}, shape).cast(dataType);
            int[] lessThan = {0, 0, 0, 0};
            int[] equalOnly = {0, 0, 0, 1};
            int[] notEqual = {1, 1, 1, 0};
            assertBools(lessThan, Tracer.trace(left, right, Tensor::lessThan), dataType, "<");
            assertBools(lessThan, Tracer.trace(left, right, Tensor::greaterThan), dataType, ">");
            assertBools(
                    equalOnly, Tracer.trace(left, right, Tensor::lessThanOrEqual), dataType, "<=");
            assertBools(
                    equalOnly,
                    Tracer.trace(left, right, Tensor::greaterThanOrEqual),
                    dataType,
                    ">=");
            assertBools(equalOnly, Tracer.trace(left, right, Tensor::equal), dataType, "==");
            assertBools(notEqual, Tracer.trace(left, right, Tensor::notEqual), dataType, "!=");
        }
    }

    private void assertBools(int[] expected, Tensor actual, DataType dataType, String op) {
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], readBool(actual, i), dataType + " " + op + " at " + i);
        }
    }

    private Tensor range(DataType dataType, Shape shape) {
        if (dataType == DataType.BOOL) {
            return boolPattern(shape);
        }
        return Tensor.iota(shape.size(), DataType.I64).cast(dataType).view(shape);
    }

    private Tensor fullZeros(DataType dataType, Shape shape) {
        return Tensor.full(0L, shape).cast(dataType);
    }

    private Tensor boolPattern(Shape shape) {
        boolean[] values = new boolean[Math.toIntExact(shape.size())];
        for (int i = 0; i < shape.size(); i++) {
            values[i] = (i % 2) == 1;
        }
        return Tensor.of(values, shape);
    }

    private byte readBool(Tensor tensor, long index) {
        return TensorTestReads.readByte(tensor, index);
    }
}
