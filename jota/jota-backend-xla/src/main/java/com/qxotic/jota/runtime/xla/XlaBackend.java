package com.qxotic.jota.runtime.xla;

import com.qxotic.jota.runtime.KernelCachePaths;
import com.qxotic.jota.runtime.xla.pjrt.PjrtApi;
import com.qxotic.jota.runtime.xla.pjrt.PjrtClient;
import com.qxotic.jota.stablehlo.portable.StableHloVersion;
import java.nio.file.Path;
import java.util.Optional;

/**
 * The process-wide XLA backend: the configured PJRT plugin and its one client, shared by every
 * {@code xla} device. Created on first use and kept for the life of the process.
 */
final class XlaBackend {

    private static XlaBackend instance;
    private static RuntimeException failure;

    private final XlaConfig config;
    private final PjrtApi api;
    private final PjrtClient client;
    private final XlaPrograms.Format format;
    private final StableHloVersion target;
    private final Path dumpDir;
    private final int cacheSize;
    private final XlaExecutableStore store;

    private XlaBackend(XlaConfig config) {
        this.config = config;
        this.format = XlaPrograms.Format.parse(config.get(XlaConfig.FORMAT, "portable"));
        this.dumpDir = config.get(XlaConfig.DUMP_DIR).map(Path::of).orElse(null);
        this.cacheSize = Integer.parseInt(config.get(XlaConfig.CACHE_SIZE, "256"));
        this.api = XlaPlugin.load(config);
        this.target = XlaPrograms.target(api);
        this.client = PjrtClient.create(api);
        boolean disk = Boolean.parseBoolean(config.get(XlaConfig.DISK_CACHE, "true"));
        this.store =
                disk && api.supports("PJRT_Executable_Serialize")
                        ? new XlaExecutableStore(
                                KernelCachePaths.programRoot(XlaRuntimeProvider.XLA)
                                        .resolve("executables"),
                                XlaExecutableStore.fingerprint(
                                        api.library(), api.version(), client.platformVersion()))
                        : XlaExecutableStore.disabled();
    }

    /**
     * The backend of this process, created from {@link XlaConfig#fromEnvironment()} on first call.
     * A failed creation is remembered and rethrown: the plugin is not loaded twice.
     */
    static synchronized XlaBackend get() {
        if (instance == null) {
            if (failure != null) {
                throw failure;
            }
            try {
                instance = new XlaBackend(XlaConfig.fromEnvironment());
            } catch (RuntimeException e) {
                failure = e;
                throw e;
            }
        }
        return instance;
    }

    /** True when a plugin is configured, without loading it. */
    static boolean isConfigured() {
        return XlaPlugin.configured(XlaConfig.fromEnvironment()).isPresent();
    }

    XlaConfig config() {
        return config;
    }

    PjrtApi api() {
        return api;
    }

    PjrtClient client() {
        return client;
    }

    XlaPrograms.Format format() {
        return format;
    }

    StableHloVersion target() {
        return target;
    }

    Optional<Path> dumpDir() {
        return Optional.ofNullable(dumpDir);
    }

    int cacheSize() {
        return cacheSize;
    }

    XlaExecutableStore store() {
        return store;
    }
}
