package com.qxotic.jota.runtime.iree.runtime;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;

/**
 * A compiled {@code .vmfb} loaded on an {@link IreeDevice}, in its own session (VM context). A
 * module compiled from a plain {@code func.func @main} is named {@code module}, so its entry point
 * is {@code module.main}.
 */
public final class IreeModule implements AutoCloseable {

    /** Entry point of a program exported by {@code TirToStableHlo.export(graph)}. */
    public static final String MAIN = "module.main";

    private final IreeDevice device;
    private final Arena arena;
    private MemorySegment session;

    IreeModule(IreeDevice device, MemorySegment session, Arena arena) {
        this.device = device;
        this.session = session;
        this.arena = arena;
    }

    /**
     * Runs {@code function} synchronously: inputs are copied to device buffers, outputs copied back
     * into new native host segments (dense, row-major).
     */
    public List<HostTensor> call(String function, List<HostTensor> inputs) {
        MemorySegment s;
        synchronized (this) {
            s = session;
        }
        if (s == null) {
            throw new IllegalStateException("IREE module is closed");
        }
        return device.call(s, function, inputs);
    }

    @Override
    public synchronized void close() {
        if (session != null) {
            device.releaseSession(session);
            session = null;
            arena.close();
        }
    }
}
