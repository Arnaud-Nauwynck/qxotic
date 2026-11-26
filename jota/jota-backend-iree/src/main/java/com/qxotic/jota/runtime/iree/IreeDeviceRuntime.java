package com.qxotic.jota.runtime.iree;

import com.qxotic.jota.Device;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.runtime.ComputeEngine;
import com.qxotic.jota.runtime.DeviceRuntime;
import com.qxotic.jota.runtime.KernelService;
import java.lang.foreign.MemorySegment;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

/**
 * The {@code iree} device: whole graphs compiled by IREE, so there is no {@link KernelService}.
 * Memory is host memory (phase A); see {@link IreeMemoryDomain}.
 */
public final class IreeDeviceRuntime implements DeviceRuntime {

    private final IreeMemoryDomain memoryDomain;
    private final IreeComputeEngine computeEngine;
    private final Map<String, String> properties;

    IreeDeviceRuntime(Device device, IreeBackend backend) {
        this.memoryDomain = new IreeMemoryDomain(device);
        this.computeEngine = new IreeComputeEngine(memoryDomain, backend.executor());
        this.properties = IreeRuntimeProvider.properties(backend.config(), backend.executor());
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
        return IreeRuntimeProvider.CAPABILITIES;
    }

    @Override
    public String toString() {
        return "DeviceRuntime{device=" + device() + ", " + properties.get("device.name") + "}";
    }
}
