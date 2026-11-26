package com.qxotic.jota.runtime.iree;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Device;
import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.memory.MemoryAllocator;
import com.qxotic.jota.memory.MemoryAllocators;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryDomains;
import com.qxotic.jota.memory.MemoryOperations;
import com.qxotic.jota.runtime.iree.runtime.IreeElementType;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Objects;

/**
 * Host memory standing for the {@code iree} device (phase A of the plan): tensors live in host
 * {@link MemorySegment}s and every execution copies its inputs into IREE buffers and its outputs
 * back. Element access and bulk operations are the host ones. Allocations are GC-managed ({@link
 * Arena#ofAuto()}).
 */
final class IreeMemoryDomain implements MemoryDomain<MemorySegment> {

    /** Host-side services (access, operations); nothing is allocated from the global arena. */
    private static final MemoryDomain<MemorySegment> HOST =
            MemoryDomains.of(MemoryAllocators.ofArena(Arena.global()));

    private final Device device;
    private final MemoryAllocator<MemorySegment> allocator;

    IreeMemoryDomain(Device device) {
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
            return new IreeMemory(
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
            return IreeElementType.isSupported(dataType);
        }
    }

    private record IreeMemory(Device device, MemorySegment base) implements Memory<MemorySegment> {

        @Override
        public long byteSize() {
            return base.byteSize();
        }

        @Override
        public boolean isReadOnly() {
            return base.isReadOnly();
        }

        @Override
        public long memoryGranularity() {
            return Byte.BYTES;
        }
    }
}
