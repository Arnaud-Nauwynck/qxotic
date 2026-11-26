package com.qxotic.jota.runtime.xla;

import com.qxotic.jota.runtime.xla.pjrt.PjrtApi;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Optional;

/** Finds and loads the PJRT plugin named by {@link XlaConfig#PLUGIN}. */
public final class XlaPlugin {

    private XlaPlugin() {}

    /** The configured plugin path, if any (not checked for existence). */
    public static Optional<XlaConfig.Value> configured(XlaConfig config) {
        return config.lookup(XlaConfig.PLUGIN);
    }

    /**
     * Loads the configured plugin.
     *
     * @throws IllegalStateException when no plugin is configured or the file does not exist, with a
     *     message naming where the setting is looked up
     */
    public static PjrtApi load(XlaConfig config) {
        XlaConfig.Value value =
                configured(config)
                        .orElseThrow(
                                () ->
                                        new IllegalStateException(
                                                "No PJRT plugin configured; set one of: "
                                                        + config.describeLookup(XlaConfig.PLUGIN)));
        Path path = Path.of(value.value());
        if (!Files.isRegularFile(path)) {
            throw new IllegalStateException(
                    "PJRT plugin "
                            + path
                            + " (from "
                            + value.describeOrigin()
                            + ") does not exist");
        }
        return PjrtApi.load(path);
    }
}
