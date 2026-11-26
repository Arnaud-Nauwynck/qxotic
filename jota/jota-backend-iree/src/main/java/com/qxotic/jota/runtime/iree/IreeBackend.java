package com.qxotic.jota.runtime.iree;

/**
 * The process-wide IREE backend: one compiler, one runtime device (the configured driver) and the
 * executor that ties them, created on first use from {@link IreeConfig#load()}.
 */
final class IreeBackend {

    private static volatile IreeBackend instance;

    private final IreeConfig config;
    private final IreeGraphExecutor executor;

    private IreeBackend(IreeConfig config) {
        this.config = config;
        this.executor = IreeGraphExecutor.fromConfig(config);
    }

    /** Whether both the compiler and the runtime library are configured (not whether they load). */
    static boolean isConfigured() {
        IreeConfig config = IreeConfig.load();
        return config.compilerLibrary().isPresent() && config.runtimeLibrary().isPresent();
    }

    /**
     * The backend, created on first call.
     *
     * @throws RuntimeException if a library is missing or fails to load (retried on the next call)
     */
    static IreeBackend get() {
        IreeBackend b = instance;
        if (b == null) {
            synchronized (IreeBackend.class) {
                b = instance;
                if (b == null) {
                    b = instance = new IreeBackend(IreeConfig.load());
                }
            }
        }
        return b;
    }

    IreeConfig config() {
        return config;
    }

    IreeGraphExecutor executor() {
        return executor;
    }
}
