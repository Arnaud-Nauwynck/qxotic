package com.qxotic.jota.runtime.iree.runtime;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.qxotic.jota.runtime.iree.IreeConfig;
import com.qxotic.jota.runtime.iree.compiler.IreeCompiler;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/** Compiles a hand-written StableHLO program and runs it on the IREE runtime through FFM. */
class IreeRuntimeTest {

    private static final String ADD_MUL =
            """
            func.func @main(%a: tensor<2x3xf32>, %b: tensor<2x3xf32>) -> (tensor<2x3xf32>, tensor<i32>) {
              %0 = stablehlo.add %a, %b : tensor<2x3xf32>
              %1 = stablehlo.multiply %0, %b : tensor<2x3xf32>
              %c = stablehlo.constant dense<42> : tensor<i32>
              return %1, %c : tensor<2x3xf32>, tensor<i32>
            }
            """;

    private static IreeConfig config() {
        IreeConfig config = IreeConfig.load();
        assumeTrue(config.compilerLibrary().isPresent(), IreeConfig.installHint());
        assumeTrue(config.runtimeLibrary().isPresent(), IreeConfig.runtimeInstallHint());
        return config;
    }

    @ParameterizedTest
    @ValueSource(strings = {"local-sync", "local-task"})
    void runsCompiledProgram(String driver) {
        IreeConfig config = config();
        byte[] vmfb =
                new IreeCompiler(
                                com.qxotic.jota.runtime.iree.compiler.IreeCompilerLibrary.open(
                                        config.compilerLibrary().get()),
                                config.compilerFlags(),
                                Optional.empty())
                        .compileText(ADD_MUL)
                        .vmfb();
        IreeRuntimeLibrary lib = IreeRuntimeLibrary.open(config.runtimeLibrary().get());
        try (IreeDevice device = IreeDevice.create(lib, driver);
                IreeModule module = device.load(vmfb);
                Arena arena = Arena.ofConfined()) {
            MemorySegment a = arena.allocateFrom(ValueLayout.JAVA_FLOAT, 1, 2, 3, 4, 5, 6);
            MemorySegment b = arena.allocateFrom(ValueLayout.JAVA_FLOAT, 1, 1, 1, 2, 2, 2);
            List<HostTensor> out =
                    module.call(
                            IreeModule.MAIN,
                            List.of(
                                    new HostTensor(IreeElementType.FLOAT_32, new long[] {2, 3}, a),
                                    new HostTensor(
                                            IreeElementType.FLOAT_32, new long[] {2, 3}, b)));
            assertEquals(2, out.size());
            assertArrayEquals(new long[] {2, 3}, out.get(0).shape());
            assertEquals(IreeElementType.FLOAT_32, out.get(0).elementType());
            assertArrayEquals(
                    new float[] {2, 3, 4, 12, 14, 16},
                    out.get(0).data().toArray(ValueLayout.JAVA_FLOAT));
            assertArrayEquals(new long[0], out.get(1).shape());
            assertEquals(42, out.get(1).data().get(ValueLayout.JAVA_INT, 0));
        }
    }

    @ParameterizedTest
    @ValueSource(strings = {"local-sync"})
    void reportsRuntimeErrors(String driver) {
        IreeConfig config = config();
        IreeRuntimeLibrary lib = IreeRuntimeLibrary.open(config.runtimeLibrary().get());
        assertThrows(IreeRuntimeException.class, () -> IreeDevice.create(lib, "no-such-driver"));
        try (IreeDevice device = IreeDevice.create(lib, driver)) {
            var e =
                    assertThrows(
                            IreeRuntimeException.class,
                            () -> device.load(new byte[] {1, 2, 3, 4, 5, 6, 7, 8}));
            assertEquals(true, e.getMessage().contains("IREE module load"), e.getMessage());
        }
    }
}
