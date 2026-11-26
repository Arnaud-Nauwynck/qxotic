package com.qxotic.jota.runtime.iree;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

/** Skips a test unless both the IREE compiler and the IREE runtime library are configured. */
final class IreeTestSupport {

    private IreeTestSupport() {}

    static IreeConfig requireCompilerAndRuntime() {
        IreeConfig config = IreeConfig.load();
        assumeTrue(
                config.compilerLibrary().isPresent(),
                "no IREE compiler configured: " + IreeConfig.installHint());
        assumeTrue(
                config.runtimeLibrary().isPresent(),
                "no IREE runtime configured: " + IreeConfig.runtimeInstallHint());
        return config;
    }
}
