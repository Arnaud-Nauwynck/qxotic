package com.qxotic.jota.runtime.iree;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.fail;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Environment;
import com.qxotic.jota.Indexing;
import com.qxotic.jota.Shape;
import com.qxotic.jota.ir.tir.TIRGraph;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.runtime.iree.compiler.IreeCompiler;
import com.qxotic.jota.runtime.iree.runtime.IreeDevice;
import com.qxotic.jota.runtime.iree.runtime.IreeRuntimeLibrary;
import com.qxotic.jota.stablehlo.portable.StableHloVersion;
import com.qxotic.jota.tensor.Tensor;
import com.qxotic.jota.tensor.TracedGraphs;
import com.qxotic.jota.tensor.Tracer;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.function.Function;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Traces jota tensor code into TIR, runs it on IREE (export, compile, load, call) and compares with
 * the panama backend's result for the same graph.
 */
class IreeGraphExecutorTest {

    @TempDir Path cacheDir;

    private IreeGraphExecutor executor(IreeGraphExecutor.Format format) {
        IreeConfig config = IreeTestSupport.requireCompilerAndRuntime();
        return new IreeGraphExecutor(
                IreeCompiler.fromConfig(config),
                IreeDevice.create(
                        IreeRuntimeLibrary.open(config.runtimeLibrary().get()), config.driver()),
                format,
                new StableHloVersion(1, 0, 0),
                Optional.of(cacheDir));
    }

    /** Runs {@code fn} traced on IREE and eagerly on panama; results must agree. */
    private void check(
            IreeGraphExecutor executor, List<Tensor> inputs, Function<List<Tensor>, Tensor> fn) {
        Tensor traced = Tracer.trace(inputs, fn);
        TIRGraph graph = TracedGraphs.graphOf(traced);
        assertEquals(inputs.size(), graph.inputCount(), "graph inputs: " + graph);
        var views = new ArrayList<MemoryView<MemorySegment>>();
        for (Tensor t : inputs) {
            views.add(segmentView(t.materialize()));
        }
        List<MemoryView<MemorySegment>> out =
                executor.execute(graph, views, Environment.nativeMemoryDomain());
        assertEquals(1, out.size());
        MemoryView<MemorySegment> expected = segmentView(traced.materialize());
        String mismatch = compare(out.getFirst(), expected);
        if (mismatch != null) {
            fail(mismatch + "\n" + new String(executor.encode(graph)));
        }
    }

    @SuppressWarnings("unchecked")
    private static MemoryView<MemorySegment> segmentView(MemoryView<?> view) {
        return (MemoryView<MemorySegment>) view;
    }

    private static Tensor f32(Shape shape, float... values) {
        return Tensor.of(values, shape);
    }

    @Test
    void elementwiseAndReductions() {
        try (var executor = executor(IreeGraphExecutor.Format.TEXT)) {
            Tensor x = f32(Shape.of(2, 3), 1f, -2f, 0.5f, 0.25f, 3f, -1f);
            Tensor w =
                    f32(Shape.of(3, 4), 0.5f, 1f, -1f, 2f, 1.5f, -0.5f, 0f, 1f, 2f, 0.25f, 1f, -1f);
            check(
                    executor,
                    List.of(x, w),
                    t -> {
                        Tensor logits = t.get(0).matmul(t.get(1));
                        Shape shape = logits.shape();
                        Tensor e = logits.subtract(logits.max(true, 1).expand(shape)).exp();
                        return e.divide(e.sum(DataType.FP32, true, 1).expand(shape));
                    });
            Tensor v = f32(Shape.of(5), -3f, -0.5f, 0f, 0.75f, 4f);
            check(
                    executor,
                    List.of(v),
                    t ->
                            t.get(0)
                                    .gelu()
                                    .add(t.get(0).silu())
                                    .add(t.get(0).sigmoid())
                                    .add(t.get(0).abs().add(1f).rsqrt()));
        }
    }

    @Test
    void integersCastsAndViews() {
        try (var executor = executor(IreeGraphExecutor.Format.TEXT)) {
            Tensor a = Tensor.of(new int[] {1, -8, 7, 1024}, Shape.of(4));
            Tensor s = Tensor.of(new int[] {1, 33, -1, 2}, Shape.of(4));
            check(
                    executor,
                    List.of(a, s),
                    t ->
                            t.get(0)
                                    .leftShift(t.get(1))
                                    .bitwiseXor(t.get(0).rightShift(t.get(1)))
                                    .bitwiseAnd(t.get(0).bitwiseNot()));
            Tensor x = f32(Shape.of(5), 0.5f, -1.5f, 0f, 2.75f, -0.5f);
            check(
                    executor,
                    List.of(x),
                    t ->
                            t.get(0)
                                    .cast(DataType.BOOL)
                                    .cast(DataType.I32)
                                    .add(t.get(0).cast(DataType.I32))
                                    .cast(DataType.FP64));
            long[] data = new long[15];
            for (int i = 0; i < data.length; i++) {
                data[i] = i;
            }
            check(
                    executor,
                    List.of(Tensor.of(data, Shape.of(3, 5))),
                    t -> {
                        Tensor x0 = t.get(0);
                        Tensor strided = x0.slice(1, 0, 5, 2);
                        Tensor tr = x0.slice(0, 2, -1, -1).transpose(0, 1).slice(0, 1, 4);
                        return strided.multiply(tr);
                    });
        }
    }

    /**
     * IREE (3.14) miscompiles {@code reduce(convert(splat constant))}, returning one element
     * instead of the sum; the exporter folds the convert into the constant.
     */
    @Test
    void sumOfConvertedSplatConstant() {
        try (var executor = executor(IreeGraphExecutor.Format.TEXT)) {
            Tensor ones = Tensor.full(1L, Shape.of(2, 3)).cast(DataType.BOOL);
            check(executor, List.of(ones), t -> t.get(0).sum(DataType.I32, 1));
            Tensor bytes = Tensor.full(1L, Shape.of(2, 3)).cast(DataType.I8);
            check(executor, List.of(bytes), t -> t.get(0).sum(DataType.I32, 1));
        }
    }

    @Test
    void stridedInputIsGathered() {
        try (var executor = executor(IreeGraphExecutor.Format.TEXT)) {
            Tensor base = f32(Shape.of(3, 4), 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
            Tensor transposed = Tensor.of(base.materialize().transpose(0, 1)); // [4, 3], strided
            check(executor, List.of(transposed), t -> t.get(0).multiply(2f).add(1f));
        }
    }

    @Test
    void portableArtifactAndCache() {
        try (var executor = executor(IreeGraphExecutor.Format.PORTABLE)) {
            Tensor x = f32(Shape.of(4), 1f, 2f, 3f, 4f);
            Function<List<Tensor>, Tensor> fn = t -> t.get(0).multiply(t.get(0)).add(1f);
            check(executor, List.of(x), fn);
            check(executor, List.of(x), fn);
            assertEquals(1, executor.compileCount(), "second run must hit the in-memory cache");
        }
        // a new executor over the same cache directory reads the .vmfb from disk
        try (var executor = executor(IreeGraphExecutor.Format.PORTABLE)) {
            check(
                    executor,
                    List.of(f32(Shape.of(4), 5f, 6f, 7f, 8f)),
                    t -> t.get(0).multiply(t.get(0)).add(1f));
            assertEquals(0, executor.compileCount(), "must hit the disk cache");
        }
    }

    /** Null when equal (exact for integers, 1e-5 relative + 1e-6 absolute for floats). */
    private static String compare(
            MemoryView<MemorySegment> actual, MemoryView<MemorySegment> expected) {
        if (actual.dataType() != expected.dataType()) {
            return "data type " + actual.dataType() + " != " + expected.dataType();
        }
        if (!actual.layout().shape().equals(expected.layout().shape())) {
            return "shape " + actual.layout().shape() + " != " + expected.layout().shape();
        }
        DataType dt = actual.dataType();
        long n = expected.layout().shape().size();
        for (long i = 0; i < n; i++) {
            double a = read(actual, i, dt);
            double e = read(expected, i, dt);
            boolean ok =
                    dt.isFloatingPoint() ? Math.abs(a - e) <= 1e-6 + 1e-5 * Math.abs(e) : a == e;
            if (!ok) {
                return "element " + i + ": IREE " + a + " != panama " + e;
            }
        }
        return null;
    }

    private static double read(MemoryView<MemorySegment> view, long i, DataType dt) {
        MemorySegment s = view.memory().base();
        long off = Indexing.linearToOffset(view, i);
        if (dt == DataType.FP32) {
            return s.get(ValueLayout.JAVA_FLOAT_UNALIGNED, off);
        } else if (dt == DataType.FP64) {
            return s.get(ValueLayout.JAVA_DOUBLE_UNALIGNED, off);
        } else if (dt == DataType.I32) {
            return s.get(ValueLayout.JAVA_INT_UNALIGNED, off);
        } else if (dt == DataType.I64) {
            return s.get(ValueLayout.JAVA_LONG_UNALIGNED, off);
        } else if (dt == DataType.I8 || dt == DataType.BOOL) {
            return s.get(ValueLayout.JAVA_BYTE, off);
        } else if (dt == DataType.I16) {
            return s.get(ValueLayout.JAVA_SHORT_UNALIGNED, off);
        }
        throw new IllegalArgumentException("unsupported in test: " + dt);
    }
}
