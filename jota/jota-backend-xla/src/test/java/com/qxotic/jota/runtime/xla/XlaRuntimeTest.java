package com.qxotic.jota.runtime.xla;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeFalse;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Device;
import com.qxotic.jota.DeviceType;
import com.qxotic.jota.Environment;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.runtime.spi.RuntimeProbeStatus;
import com.qxotic.jota.runtime.xla.pjrt.CompileOptions;
import com.qxotic.jota.stablehlo.attr.DenseElementsAttr;
import com.qxotic.jota.stablehlo.tir.TirConversion;
import com.qxotic.jota.stablehlo.type.FloatType;
import com.qxotic.jota.stablehlo.type.TensorType;
import com.qxotic.jota.tensor.Tensor;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.function.Function;
import org.junit.jupiter.api.Test;

/**
 * The {@code xla} device through jota's runtime registry. The plugin-free tests always run; the
 * execution tests need a configured plugin (see {@link XlaConfig}).
 */
class XlaRuntimeTest {

    private static final Device XLA = XlaRuntimeProvider.XLA.deviceIndex(0);
    private static final Device PANAMA = DeviceType.PANAMA.deviceIndex(0);

    @Test
    void compileOptionsWireFormat() {
        // CompileOptionsProto{executable_build_options{num_replicas: 1, num_partitions: 1}}
        assertArrayEquals(new byte[] {0x1A, 4, 0x20, 1, 0x28, 1}, CompileOptions.singleDevice());
        // ... plus device_assignment{replica_count: 1, computation_count: 1,
        //     computation_devices{replica_device_ids: 3}}
        assertArrayEquals(
                new byte[] {
                    0x1A, 14, 0x20, 1, 0x28, 1, 0x4A, 8, 0x08, 1, 0x10, 1, 0x1A, 2, 0x08, 3
                },
                CompileOptions.singleDevice(3));
    }

    @Test
    void unavailableWithoutPlugin() {
        assumeFalse(XlaBackend.isConfigured(), "a PJRT plugin is configured");
        var probe = new XlaRuntimeProvider().probe();
        assertEquals(RuntimeProbeStatus.MISSING_SOFTWARE, probe.status());
        assertTrue(probe.hint().contains("JOTA_XLA_PLUGIN"), probe.hint());
        assertEquals(List.of(), new XlaRuntimeProvider().availableDevices());
    }

    @Test
    void runsTensorProgramsOnXla() {
        assumeTrue(XlaBackend.isConfigured(), "no PJRT plugin configured");
        assertTrue(
                Environment.hasRuntimeFor(XLA), () -> Environment.runtimeDiagnostics().toString());

        Tensor x = Tensor.of(new float[] {-3f, -0.5f, 0f, 0.75f, 4f, 1.25f}, Shape.of(2, 3));
        Tensor w = Tensor.of(new float[] {0.5f, 1f, -1f, 2f, 1.5f, -0.5f}, Shape.of(3, 2));
        agree(List.of(x, w), t -> t.get(0).matmul(t.get(1)).tanh());
        agree(List.of(x), t -> t.get(0).gelu().add(t.get(0).silu()).multiply(Tensor.scalar(2f)));
        agree(List.of(x), t -> t.get(0).sum(DataType.FP32, 1).add(t.get(0).max(1)));
        // a strided (transposed) input: copied to row-major before the upload
        agree(List.of(x.transpose(0, 1)), t -> t.get(0).exp());
        // chaining: the result of one XLA execution feeds the next
        agree(List.of(x, w), t -> t.get(0).matmul(t.get(1)).matmul(t.get(0)).sqrt().abs());
    }

    @Test
    void chainedResultsStayOnDevice() {
        assumeTrue(XlaBackend.isConfigured(), "no PJRT plugin configured");
        Tensor x = Tensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f}, Shape.of(2, 3));
        Tensor result =
                Environment.with(
                        Environment.withDefaultDevice(XLA),
                        () -> {
                            Tensor a = x.to(XLA);
                            long uploads = XlaMemory.UPLOADS.get();
                            long reuses = XlaMemory.REUSES.get();
                            long downloads = XlaMemory.DOWNLOADS.get();
                            Tensor t = a.exp();
                            t.materialize();
                            for (int i = 0; i < 4; i++) {
                                t = t.add(t).multiply(Tensor.scalar(0.5f)).sqrt();
                                t.materialize();
                            }
                            System.out.printf(
                                    "chain: uploads %d, device reuses %d, downloads %d%n",
                                    XlaMemory.UPLOADS.get() - uploads,
                                    XlaMemory.REUSES.get() - reuses,
                                    XlaMemory.DOWNLOADS.get() - downloads);
                            assertTrue(
                                    XlaMemory.REUSES.get() - reuses > 0,
                                    "results should feed the next execution on device");
                            assertEquals(
                                    downloads,
                                    XlaMemory.DOWNLOADS.get(),
                                    "nothing should be copied back before the host reads");
                            return t;
                        });
        long downloads = XlaMemory.DOWNLOADS.get();
        float[] values = floats(result);
        assertEquals(downloads + 1, XlaMemory.DOWNLOADS.get());
        // each step is sqrt((t + t) * 0.5) = sqrt(t): four steps give exp(x)^(1/16)
        for (int i = 0; i < values.length; i++) {
            double expected = Math.pow(Math.exp(i + 1), 1.0 / 16);
            assertEquals(expected, values[i], 1e-5 * expected, "element " + i);
        }
    }

    /** Runs {@code fn} on panama and on xla with the same inputs, and compares. */
    private static void agree(List<Tensor> inputs, Function<List<Tensor>, Tensor> fn) {
        float[] expected = floats(onDevice(PANAMA, inputs, fn));
        Tensor onXla = onDevice(XLA, inputs, fn);
        assertEquals(XLA, onXla.device());
        float[] actual = floats(onXla);
        assertEquals(expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) {
            if (Float.isNaN(expected[i])) {
                assertTrue(Float.isNaN(actual[i]), "element " + i);
                continue;
            }
            assertEquals(
                    expected[i],
                    actual[i],
                    1e-5f * Math.max(1f, Math.abs(expected[i])),
                    "element " + i);
        }
    }

    private static Tensor onDevice(
            Device device, List<Tensor> inputs, Function<List<Tensor>, Tensor> fn) {
        return Environment.with(
                Environment.withDefaultDevice(device),
                () -> {
                    List<Tensor> moved = inputs.stream().map(t -> t.to(device)).toList();
                    Tensor result = fn.apply(moved);
                    result.materialize();
                    return result;
                });
    }

    @SuppressWarnings("unchecked")
    private static float[] floats(Tensor t) {
        Tensor host = t.to(PANAMA);
        DenseElementsAttr d =
                TirConversion.toDense(
                        (MemoryView<MemorySegment>) host.materialize(),
                        TensorType.of(FloatType.F32, host.shape().flattenModes().toArray()));
        float[] out = new float[(int) d.numElements()];
        for (int i = 0; i < out.length; i++) {
            out[i] = (float) d.getDouble(i);
        }
        return out;
    }
}
