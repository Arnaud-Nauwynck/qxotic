package com.qxotic.jota.runtime.xla;

import com.qxotic.jota.Device;
import com.qxotic.jota.DeviceType;
import com.qxotic.jota.runtime.DeviceRuntime;
import com.qxotic.jota.runtime.spi.DeviceRuntimeProvider;
import com.qxotic.jota.runtime.spi.RuntimeProbe;
import com.qxotic.jota.runtime.xla.pjrt.PjrtDevice;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

/**
 * Provides the {@code xla} devices: the addressable devices of the PJRT plugin named by {@link
 * XlaConfig#PLUGIN}. Unavailable (missing software) when no plugin is configured.
 */
public final class XlaRuntimeProvider extends DeviceRuntimeProvider {

    /** The {@code xla} device type. */
    public static final DeviceType XLA = DeviceType.fromId("xla");

    static final Set<String> CAPABILITIES = Set.of("graph.compilation", "fp32", "fp64");

    @Override
    public DeviceType deviceType() {
        return XLA;
    }

    @Override
    public RuntimeProbe probe() {
        if (!XlaBackend.isConfigured()) {
            XlaConfig config = XlaConfig.fromEnvironment();
            return RuntimeProbe.missingSoftware(
                    "No PJRT plugin configured for the XLA backend",
                    "Set the plugin library with one of: "
                            + config.describeLookup(XlaConfig.PLUGIN));
        }
        try {
            XlaBackend backend = XlaBackend.get();
            int count = backend.client().devices().size();
            if (count == 0) {
                return RuntimeProbe.unsupportedHardware(
                        "PJRT plugin " + backend.api().library() + " has no addressable device",
                        "Check that the plugin's hardware is present and visible");
            }
            return RuntimeProbe.available(
                    "XLA "
                            + backend.client().platformName()
                            + " via "
                            + backend.api()
                            + " with "
                            + count
                            + " device(s)");
        } catch (RuntimeException | LinkageError e) {
            return RuntimeProbe.misconfigured(
                    "PJRT plugin failed to load: " + e.getMessage(),
                    "Check jota.xla.plugin and that the plugin matches this machine",
                    e);
        }
    }

    @Override
    public int deviceCount() {
        try {
            return XlaBackend.get().client().devices().size();
        } catch (RuntimeException e) {
            return 0;
        }
    }

    @Override
    public Map<String, String> properties(int deviceIndex) {
        try {
            XlaBackend backend = XlaBackend.get();
            return properties(backend, backend.client().devices().get(deviceIndex));
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
        return new XlaDeviceRuntime(device, XlaBackend.get());
    }

    static Map<String, String> properties(XlaBackend backend, PjrtDevice device) {
        var props = new LinkedHashMap<String, String>();
        props.put("device.name", backend.client().platformName() + ":" + device.kind());
        props.put("device.vendor", "XLA");
        props.put("device.kind", device.kind());
        props.put("device.driver.version", backend.client().platformVersion());
        props.put("pjrt.plugin", backend.api().library().toString());
        props.put("pjrt.api.version", backend.api().version());
        props.put("pjrt.device.id", Integer.toString(device.id()));
        props.put("stablehlo.target", backend.target().toString());
        props.put("xla.format", backend.format().name().toLowerCase(java.util.Locale.ROOT));
        return Map.copyOf(props);
    }
}
