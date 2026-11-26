package com.qxotic.jota.runtime.xla;

import com.qxotic.jota.Device;
import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.runtime.xla.pjrt.PjrtBuffer;
import java.lang.foreign.MemorySegment;
import java.lang.ref.Cleaner;
import java.util.Arrays;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Memory of an {@code xla} device: a host segment, plus optionally the PJRT buffer that holds the
 * same elements on the device (a "device mirror").
 *
 * <p>An execution result keeps its buffer as mirror and leaves the host segment unfilled; the next
 * execution that reads the whole memory passes the buffer on without any transfer. The host segment
 * is brought up to date the first time {@link #base()} is called; since the caller may then write
 * through it, the mirror is dropped at that point (a later execution uploads again).
 *
 * <p>{@link #LOCK} orders executions against mirror changes; unreachable memories release their
 * buffer through a {@link Cleaner}.
 */
final class XlaMemory implements Memory<MemorySegment> {

    /** Held by executions (which read mirrors) and by every change of a mirror. */
    static final ReentrantLock LOCK = new ReentrantLock();

    private static final Cleaner CLEANER = Cleaner.create();

    /** Transfer counters, for tests and diagnostics. */
    static final AtomicLong UPLOADS = new AtomicLong();

    static final AtomicLong REUSES = new AtomicLong();
    static final AtomicLong DOWNLOADS = new AtomicLong();

    private final Device device;
    private final MemorySegment host;
    private final Mirror mirror = new Mirror();

    /** The cleaner's state: must not reference the memory. */
    private static final class Mirror implements Runnable {
        PjrtBuffer buffer;
        long[] dims;
        int deviceId;
        boolean hostStale;

        void clear() {
            if (buffer != null) {
                buffer.close();
            }
            buffer = null;
            dims = null;
            hostStale = false;
        }

        @Override
        public void run() {
            LOCK.lock();
            try {
                clear();
            } finally {
                LOCK.unlock();
            }
        }
    }

    XlaMemory(Device device, MemorySegment host) {
        this.device = Objects.requireNonNull(device, "device");
        this.host = Objects.requireNonNull(host, "host");
        CLEANER.register(this, mirror);
    }

    @Override
    public Device device() {
        return device;
    }

    /** The host segment, downloaded first if a device result has not been copied yet. */
    @Override
    public MemorySegment base() {
        LOCK.lock();
        try {
            if (mirror.buffer != null) {
                if (mirror.hostStale) {
                    mirror.buffer.copyToHost(host);
                    DOWNLOADS.incrementAndGet();
                }
                mirror.clear();
            }
            return host;
        } finally {
            LOCK.unlock();
        }
    }

    @Override
    public long byteSize() {
        return host.byteSize();
    }

    @Override
    public boolean isReadOnly() {
        return host.isReadOnly();
    }

    @Override
    public long memoryGranularity() {
        return Byte.BYTES;
    }

    /**
     * Makes {@code buffer} (already on device {@code deviceId}, with {@code dims}) this memory's
     * contents; the host segment is stale until the next {@link #base()}. Takes ownership.
     */
    void setDeviceContents(PjrtBuffer buffer, long[] dims, int deviceId) {
        LOCK.lock();
        try {
            mirror.clear();
            mirror.buffer = buffer;
            mirror.dims = dims.clone();
            mirror.deviceId = deviceId;
            mirror.hostStale = true;
        } finally {
            LOCK.unlock();
        }
    }

    /**
     * The mirror when it holds exactly {@code dims} on {@code deviceId}, else null. The caller must
     * hold {@link #LOCK} for as long as it uses the buffer, and must not close it.
     */
    PjrtBuffer deviceBuffer(long[] dims, int deviceId) {
        if (!LOCK.isHeldByCurrentThread()) {
            throw new IllegalStateException("XlaMemory.LOCK not held");
        }
        if (mirror.buffer == null
                || mirror.deviceId != deviceId
                || !Arrays.equals(mirror.dims, dims)) {
            return null;
        }
        return mirror.buffer;
    }

    /** True while a device result has not been copied to the host. */
    boolean isHostStale() {
        LOCK.lock();
        try {
            return mirror.buffer != null && mirror.hostStale;
        } finally {
            LOCK.unlock();
        }
    }
}
