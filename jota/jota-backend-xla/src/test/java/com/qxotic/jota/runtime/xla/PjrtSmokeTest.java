package com.qxotic.jota.runtime.xla;

import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.runtime.xla.pjrt.CompileOptions;
import com.qxotic.jota.runtime.xla.pjrt.PjrtApi;
import com.qxotic.jota.runtime.xla.pjrt.PjrtBuffer;
import com.qxotic.jota.runtime.xla.pjrt.PjrtBufferType;
import com.qxotic.jota.runtime.xla.pjrt.PjrtClient;
import com.qxotic.jota.runtime.xla.pjrt.PjrtDevice;
import com.qxotic.jota.runtime.xla.pjrt.PjrtException;
import com.qxotic.jota.runtime.xla.pjrt.PjrtLoadedExecutable;
import com.qxotic.jota.stablehlo.attr.DenseElementsAttr;
import com.qxotic.jota.stablehlo.ir.Program;
import com.qxotic.jota.stablehlo.portable.StableHloVersion;
import com.qxotic.jota.stablehlo.tir.TirConversion;
import com.qxotic.jota.stablehlo.tir.TirToStableHlo;
import com.qxotic.jota.stablehlo.type.FloatType;
import com.qxotic.jota.stablehlo.type.TensorType;
import com.qxotic.jota.tensor.Tensor;
import com.qxotic.jota.tensor.TracedGraphs;
import com.qxotic.jota.tensor.Tracer;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

/**
 * Drives a real PJRT plugin through the FFM bindings. Skipped unless a plugin is configured ({@code
 * -Djota.xla.plugin=...}, {@code JOTA_XLA_PLUGIN}, or a jota.properties file; see {@link
 * XlaConfig}).
 */
class PjrtSmokeTest {

    private static PjrtApi api;
    private static PjrtClient client;

    @BeforeAll
    static void open() {
        XlaConfig config = XlaConfig.fromEnvironment();
        assumeTrue(
                XlaPlugin.configured(config).isPresent(),
                "no PJRT plugin configured: " + config.describeLookup(XlaConfig.PLUGIN));
        api = XlaPlugin.load(config);
        client = PjrtClient.create(api);
    }

    @AfterAll
    static void close() {
        if (client != null) {
            client.close();
        }
    }

    @Test
    void describesPluginAndDevices() {
        assertEquals(PjrtApi.MAJOR, api.majorVersion());
        assertTrue(api.supports("PJRT_Client_Compile"));
        assertFalse(client.platformName().isEmpty());
        assertFalse(client.devices().isEmpty());
        System.out.println(
                api
                        + ", platform "
                        + client.platformName()
                        + " "
                        + client.platformVersion()
                        + ", "
                        + client.devices().size()
                        + " device(s) of kind "
                        + client.devices().get(0).kind());
        api.attributes()
                .forEach((k, v) -> System.out.println("  " + k + " = " + PjrtApi.format(v)));
    }

    @Test
    void runsHandWrittenStableHlo() {
        String text =
                """
                func.func @main(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>) -> tensor<2x2xf32> {
                  %0 = stablehlo.add %a, %b : tensor<2x2xf32>
                  %1 = stablehlo.multiply %0, %b : tensor<2x2xf32>
                  return %1 : tensor<2x2xf32>
                }
                """;
        float[] out =
                run(
                        text.getBytes(StandardCharsets.UTF_8),
                        new float[][] {{1, 2, 3, 4}, {10, 20, 30, 40}},
                        new long[] {2, 2});
        assertArrayEquals(new float[] {110, 440, 990, 1760}, out);
    }

    @Test
    void serializedExecutablesLoadAgain() {
        assumeTrue(api.supports("PJRT_Executable_Serialize"), "plugin cannot serialize");
        String text =
                """
                func.func @main(%a: tensor<3xf32>) -> tensor<3xf32> {
                  %0 = stablehlo.multiply %a, %a : tensor<3xf32>
                  return %0 : tensor<3xf32>
                }
                """;
        byte[] serialized;
        try (PjrtLoadedExecutable exe =
                client.compile(
                        text.getBytes(StandardCharsets.UTF_8),
                        XlaPrograms.PJRT_FORMAT,
                        CompileOptions.singleDevice())) {
            serialized = exe.serialize();
        }
        assertTrue(serialized.length > 0);
        try (PjrtLoadedExecutable loaded =
                client.deserialize(serialized, CompileOptions.singleDevice())) {
            float[] out = run(loaded, new float[][] {{1, -2, 3}}, new long[][] {{3}});
            assertArrayEquals(new float[] {1, 4, 9}, out);
        }
    }

    @Test
    void compileErrorsBecomeExceptions() {
        PjrtException e =
                assertThrows(
                        PjrtException.class,
                        () ->
                                client.compile(
                                        "this is not mlir".getBytes(StandardCharsets.UTF_8),
                                        XlaPrograms.PJRT_FORMAT,
                                        CompileOptions.singleDevice()));
        assertEquals("PJRT_Client_Compile", e.function());
        assertFalse(e.getMessage().isBlank());
    }

    @ParameterizedTest
    @EnumSource(XlaPrograms.Format.class)
    void runsExportedJotaGraphs(XlaPrograms.Format format) {
        StableHloVersion target = XlaPrograms.target(api);
        Tensor x = Tensor.of(new float[] {-3f, -0.5f, 0f, 0.75f, 4f, 1.25f}, Shape.of(2, 3));
        Tensor w = Tensor.of(new float[] {0.5f, 1f, -1f, 2f, 1.5f, -0.5f}, Shape.of(3, 2));
        check(format, target, List.of(x, w), t -> t.get(0).matmul(t.get(1)).tanh());
        check(
                format,
                target,
                List.of(x),
                t -> t.get(0).gelu().add(t.get(0).silu()).add(t.get(0).sigmoid()));
        check(
                format,
                target,
                List.of(x),
                t -> t.get(0).transpose(0, 1).exp().sum(DataType.FP32, 1));
    }

    private void check(
            XlaPrograms.Format format,
            StableHloVersion target,
            List<Tensor> inputs,
            Function<List<Tensor>, Tensor> fn) {
        Tensor traced = Tracer.trace(inputs, fn);
        Program program = TirToStableHlo.export(TracedGraphs.graphOf(traced));
        byte[] code = XlaPrograms.encode(program, format, target);
        float[][] in = new float[inputs.size()][];
        long[][] dims = new long[inputs.size()][];
        for (int i = 0; i < inputs.size(); i++) {
            in[i] = floats(inputs.get(i));
            dims[i] = inputs.get(i).shape().flattenModes().toArray();
        }
        float[] expected = floats(traced);
        float[] actual = run(code, in, dims);
        assertEquals(expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) {
            assertEquals(
                    expected[i],
                    actual[i],
                    1e-5f * Math.max(1f, Math.abs(expected[i])),
                    format
                            + " element "
                            + i
                            + "\n"
                            + new String(
                                    XlaPrograms.encode(program, XlaPrograms.Format.TEXT, target),
                                    StandardCharsets.UTF_8));
        }
    }

    private float[] run(byte[] code, float[][] inputs, long[] dims) {
        long[][] all = new long[inputs.length][];
        java.util.Arrays.fill(all, dims);
        return run(code, inputs, all);
    }

    private float[] run(byte[] code, float[][] inputs, long[][] dims) {
        try (PjrtLoadedExecutable exe =
                client.compile(code, XlaPrograms.PJRT_FORMAT, CompileOptions.singleDevice())) {
            return run(exe, inputs, dims);
        }
    }

    private float[] run(PjrtLoadedExecutable exe, float[][] inputs, long[][] dims) {
        PjrtDevice device = client.devices().get(0);
        try (Arena arena = Arena.ofConfined()) {
            assertEquals(1, exe.numOutputs());
            List<PjrtBuffer> args = new ArrayList<>();
            try {
                for (int i = 0; i < inputs.length; i++) {
                    MemorySegment host = arena.allocateFrom(JAVA_FLOAT, inputs[i]);
                    args.add(client.bufferFromHost(host, PjrtBufferType.F32, dims[i], device));
                }
                List<PjrtBuffer> outs = exe.execute(args);
                try (PjrtBuffer out = outs.get(0)) {
                    assertEquals(PjrtBufferType.F32, out.type());
                    return out.toHost(arena).toArray(JAVA_FLOAT);
                }
            } finally {
                args.forEach(PjrtBuffer::close);
            }
        }
    }

    @SuppressWarnings("unchecked")
    private static float[] floats(Tensor t) {
        DenseElementsAttr d =
                TirConversion.toDense(
                        (MemoryView<MemorySegment>) t.materialize(),
                        TensorType.of(FloatType.F32, t.shape().flattenModes().toArray()));
        float[] out = new float[(int) d.numElements()];
        for (int i = 0; i < out.length; i++) {
            out[i] = (float) d.getDouble(i);
        }
        return out;
    }
}
