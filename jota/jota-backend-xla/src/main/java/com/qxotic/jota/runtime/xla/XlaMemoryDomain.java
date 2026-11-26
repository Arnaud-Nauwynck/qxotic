package com.qxotic.jota.runtime.xla;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Device;
import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.memory.MemoryAllocator;
import com.qxotic.jota.memory.MemoryAllocators;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryDomains;
import com.qxotic.jota.memory.MemoryOperations;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Objects;

/**
 * Host memory standing for an XLA device (phase A of the plan): tensors of the {@code xla} device
 * live in host {@link MemorySegment}s, and every execution copies its inputs to the PJRT device and
 * its result back. Element access and bulk operations are the host ones.
 *
 * <p>Allocations are GC-managed ({@link Arena#ofAuto()}): a memory is freed once unreachable.
 */
final class XlaMemoryDomain implements MemoryDomain<MemorySegment> {

    /** Host-side services (access, operations); the global arena is never allocated from here. */
    private static final MemoryDomain<MemorySegment> HOST =
            MemoryDomains.of(MemoryAllocators.ofArena(Arena.global()));

    private final Device device;
    private final MemoryAllocator<MemorySegment> allocator;

    XlaMemoryDomain(Device device) {
        this.device = Objects.requireNonNull(device, "device");
        this.allocator = new Allocator(device);
    }

    @Override
    public Device device() {
        return device;
    }

    @Override
    public MemoryAllocator<MemorySegment> memoryAllocator() {
        return allocator;
    }

    @Override
    public MemoryAccess<MemorySegment> directAccess() {
        return HOST.directAccess();
    }

    @Override
    public MemoryOperations<MemorySegment> memoryOperations() {
        return HOST.memoryOperations();
    }

    @Override
    public void close() {}

    private record Allocator(Device device) implements MemoryAllocator<MemorySegment> {

        @Override
        public Memory<MemorySegment> allocateMemory(long byteSize, long byteAlignment) {
            if (byteSize < 0) {
                throw new IllegalArgumentException("invalid byteSize, must be >= 0");
            }
            if (byteAlignment <= 0 || (byteAlignment & (byteAlignment - 1)) != 0) {
                throw new IllegalArgumentException("invalid byteAlignment, not a power of 2");
            }
            return new XlaMemory(
                    device, Arena.ofAuto().allocate(byteSize, Math.max(byteAlignment, 64)));
        }

        @Override
        public long defaultByteAlignment() {
            return 64;
        }

        @Override
        public long memoryGranularity() {
            return Byte.BYTES;
        }

        @Override
        public boolean supportsDataType(DataType dataType) {
            return XlaDataTypes.isSupported(dataType);
        }
    }
}
