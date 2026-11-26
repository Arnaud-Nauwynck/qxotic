package com.qxotic.jota.runtime.iree.compiler;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.qxotic.jota.runtime.iree.IreeConfig;
import com.qxotic.jota.runtime.iree.JotaConfig;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/**
 * Compiles real programs when an IREE compiler is configured ({@code -Djota.iree.compiler.lib},
 * {@code JOTA_IREE_COMPILER_LIB}, {@code config/jota.properties} or {@code
 * ~/jota/jota.properties}); skipped otherwise.
 */
class IreeCompilerTest {

    private static final String ADD =
            """
            func.func @main(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
              %0 = stablehlo.add %a, %b : tensor<4xf32>
              return %0 : tensor<4xf32>
            }
            """;

    private static IreeCompiler compiler(String backends) {
        IreeConfig config = IreeConfig.load();
        assumeTrue(
                config.compilerLibrary().isPresent(),
                "no IREE compiler configured: " + IreeConfig.installHint());
        IreeCompilerLibrary library = IreeCompilerLibrary.open(config.compilerLibrary().get());
        var flags =
                List.of(
                        "--iree-hal-target-device=local",
                        "--iree-hal-local-target-device-backends=" + backends);
        return new IreeCompiler(library, flags, config.dumpDir());
    }

    @Test
    void bytecodeMagicDetection() {
        assertTrue(IreeCompiler.isMlirBytecode(new byte[] {'M', 'L', (byte) 0xEF, 'R', 0}));
        assertFalse(IreeCompiler.isMlirBytecode("module {}".getBytes()));
        assertFalse(IreeCompiler.isMlirBytecode(new byte[0]));
    }

    @Test
    void libraryReportsVersionAndTargets() {
        IreeCompilerLibrary library = compiler("vmvx").library();
        System.out.println(
                "IREE compiler "
                        + library.path()
                        + " api="
                        + library.apiVersionString()
                        + " revision="
                        + library.revision()
                        + " origin="
                        + JotaConfig.global()
                                .lookup(IreeConfig.COMPILER_LIB)
                                .map(JotaConfig.Value::describeOrigin)
                                .orElse("?"));
        assertTrue(library.registeredHalTargetBackends().contains("vmvx"));
    }

    @Test
    void compilesStableHloAddForVmvx() {
        var module = compiler("vmvx").compileText(ADD);
        assertTrue(module.hasVmfbIdentifier(), "not a .vmfb");
    }

    @Test
    void compilesStableHloAddForLlvmCpu() {
        IreeCompiler base = compiler("llvm-cpu");
        var flags = new java.util.ArrayList<>(base.flags());
        flags.add("--iree-llvmcpu-target-cpu=host");
        var module = new IreeCompiler(base.library(), flags, Optional.empty()).compileText(ADD);
        assertTrue(module.hasVmfbIdentifier(), "not a .vmfb");
    }

    /**
     * LLVM installs its own SIGSEGV / SIGBUS / ... handlers during an llvm-cpu compile; the JVM
     * needs its own back (it uses SIGSEGV for null checks and safepoints) or it dies on the next
     * implicit null check.
     */
    @Test
    void jvmSignalHandlersSurviveLlvmCpuCompile() {
        org.junit.jupiter.api.Assumptions.assumeFalse(
                System.getProperty("os.name", "").toLowerCase().contains("win"));
        IreeCompiler base = compiler("llvm-cpu"); // loads the library: handlers captured before
        int[] signals = {4, 7, 8, 11}; // SIGILL, SIGBUS (Linux), SIGFPE, SIGSEGV
        long[] before = new long[signals.length];
        for (int i = 0; i < signals.length; i++) {
            before[i] = SignalHandlerGuard.currentHandler(signals[i]);
        }
        var flags = new java.util.ArrayList<>(base.flags());
        flags.add("--iree-llvmcpu-target-cpu=host");
        new IreeCompiler(base.library(), flags, Optional.empty()).compileText(ADD);
        for (int i = 0; i < signals.length; i++) {
            org.junit.jupiter.api.Assertions.assertEquals(
                    before[i],
                    SignalHandlerGuard.currentHandler(signals[i]),
                    "handler of signal " + signals[i] + " replaced by the compiler");
        }
    }

    @Test
    void invalidProgramReportsDiagnostics() {
        IreeCompiler compiler = compiler("vmvx");
        var e =
                assertThrows(
                        IreeCompilerException.class,
                        () -> compiler.compileText("func.func @main( {"));
        assertFalse(e.diagnostics().isEmpty(), "expected diagnostics: " + e.getMessage());
    }
}
