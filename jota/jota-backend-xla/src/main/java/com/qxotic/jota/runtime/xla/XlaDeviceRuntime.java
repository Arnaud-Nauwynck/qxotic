package com.qxotic.jota.runtime.xla;

import com.qxotic.jota.Device;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.runtime.ComputeEngine;
import com.qxotic.jota.runtime.DeviceRuntime;
import com.qxotic.jota.runtime.KernelService;
import com.qxotic.jota.runtime.xla.pjrt.PjrtDevice;
import java.lang.foreign.MemorySegment;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

/**
 * An {@code xla} device: one addressable device of the configured PJRT plugin. XLA compiles whole
 * graphs, so there is no {@link KernelService}.
 */
public final class XlaDeviceRuntime implements DeviceRuntime {

    private final XlaMemoryDomain memoryDomain;
    private final XlaComputeEngine computeEngine;
    private final Map<String, String> properties;

    XlaDeviceRuntime(Device device, XlaBackend backend) {
        int index = Math.toIntExact(device.index());
        var devices = backend.client().devices();
        if (index < 0 || index >= devices.size()) {
            throw new IllegalArgumentException(
                    "No XLA device " + index + ": the plugin has " + devices.size());
        }
        PjrtDevice pjrtDevice = devices.get(index);
        this.memoryDomain = new XlaMemoryDomain(device);
        this.computeEngine = new XlaComputeEngine(backend, memoryDomain, pjrtDevice);
        this.properties = XlaRuntimeProvider.properties(backend, pjrtDevice);
    }

    @Override
    public Device device() {
        return memoryDomain.device();
    }

    @Override
    public MemoryDomain<MemorySegment> memoryDomain() {
        return memoryDomain;
    }

    @Override
    public ComputeEngine computeEngine() {
        return computeEngine;
    }

    @Override
    public Optional<KernelService> kernelService() {
        return Optional.empty();
    }

    @Override
    public Map<String, String> properties() {
        return properties;
    }

    @Override
    public Set<String> capabilities() {
        return XlaRuntimeProvider.CAPABILITIES;
    }

    @Override
    public String toString() {
        return "DeviceRuntime{device=" + device() + ", " + properties.get("device.name") + "}";
    }
}
