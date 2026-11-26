package com.qxotic.jota.runtime.iree;

import com.qxotic.jota.Device;
import com.qxotic.jota.DeviceType;
import com.qxotic.jota.runtime.DeviceRuntime;
import com.qxotic.jota.runtime.spi.DeviceRuntimeProvider;
import com.qxotic.jota.runtime.spi.RuntimeProbe;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

/**
 * Provides the {@code iree} device: TIR graphs compiled by the IREE compiler and run by the IREE
 * runtime on the configured driver. Unavailable (missing software) until both {@link
 * IreeConfig#COMPILER_LIB} and {@link IreeConfig#RUNTIME_LIB} are configured.
 */
public final class IreeRuntimeProvider extends DeviceRuntimeProvider {

    /** The {@code iree} device type. */
    public static final DeviceType IREE = DeviceType.fromId("iree");

    static final Set<String> CAPABILITIES = Set.of("graph.compilation", "fp32", "fp64");

    @Override
    public DeviceType deviceType() {
        return IREE;
    }

    @Override
    public RuntimeProbe probe() {
        IreeConfig config = IreeConfig.load();
        if (config.compilerLibrary().isEmpty()) {
            return RuntimeProbe.missingSoftware(
                    "No IREE compiler configured for the IREE backend", IreeConfig.installHint());
        }
        if (config.runtimeLibrary().isEmpty()) {
            return RuntimeProbe.missingSoftware(
                    "No IREE runtime library configured for the IREE backend",
                    IreeConfig.runtimeInstallHint());
        }
        try {
            IreeBackend backend = IreeBackend.get();
            return RuntimeProbe.available(
                    "IREE "
                            + backend.executor().compilerRevision()
                            + " on "
                            + backend.config().driver()
                            + " (target "
                            + backend.config().targetDevice()
                            + "/"
                            + backend.config().targetBackends()
                            + ")");
        } catch (RuntimeException | LinkageError e) {
            return RuntimeProbe.misconfigured(
                    "IREE backend failed to load: " + e.getMessage(),
                    "Check "
                            + IreeConfig.COMPILER_LIB
                            + ", "
                            + IreeConfig.RUNTIME_LIB
                            + " (same IREE revision) and "
                            + IreeConfig.DRIVER,
                    e);
        }
    }

    @Override
    public Map<String, String> properties(int deviceIndex) {
        try {
            IreeBackend backend = IreeBackend.get();
            return properties(backend.config(), backend.executor());
        } catch (RuntimeException e) {
            return Map.of();
        }
    }

    @Override
    public Set<String> capabilities(int deviceIndex) {
        return CAPABILITIES;
    }

    @Override
    protected DeviceRuntime createForDevice(Device device) {
        if (device.index() != 0) {
            throw new IllegalArgumentException("No IREE device " + device.index() + ": only 0");
        }
        return new IreeDeviceRuntime(device, IreeBackend.get());
    }

    static Map<String, String> properties(IreeConfig config, IreeGraphExecutor executor) {
        var props = new LinkedHashMap<String, String>();
        props.put("device.name", "IREE " + config.driver());
        props.put("device.vendor", "IREE");
        props.put("device.kind", config.targetDevice().equals("local") ? "cpu" : "gpu");
        props.put("device.driver.version", executor.compilerRevision());
        props.put("iree.driver", config.driver());
        props.put("iree.target.device", config.targetDevice());
        props.put("iree.target.backends", config.targetBackends());
        return Map.copyOf(props);
    }
}
