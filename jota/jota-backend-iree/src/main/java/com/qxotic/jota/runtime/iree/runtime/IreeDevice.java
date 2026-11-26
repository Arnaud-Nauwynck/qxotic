package com.qxotic.jota.runtime.iree.runtime;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * An IREE runtime instance with one HAL device, e.g. {@code local-task} (multi-threaded CPU),
 * {@code local-sync} (CPU, calling thread) or {@code cuda}. Compiled modules are loaded on it with
 * {@link #load(byte[])}.
 *
 * <p>Thread-safe: native calls on the device and its modules are serialized on this object.
 */
public final class IreeDevice implements AutoCloseable {

    private final IreeRuntimeLibrary lib;
    private final String uri;
    private MemorySegment instance;
    private MemorySegment device;

    private IreeDevice(IreeRuntimeLibrary lib, String uri) {
        this.lib = lib;
        this.uri = uri;
    }

    /** Creates an instance with every compiled-in driver and the device named by {@code uri}. */
    public static IreeDevice create(IreeRuntimeLibrary lib, String uri) {
        Objects.requireNonNull(lib, "lib");
        Objects.requireNonNull(uri, "uri");
        var d = new IreeDevice(lib, uri);
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment out = arena.allocate(ADDRESS);
            lib.check((MemorySegment) lib.instanceCreate.invokeExact(out), "IREE instance create");
            d.instance = out.get(ADDRESS, 0);
            lib.check(
                    (MemorySegment)
                            lib.deviceCreate.invokeExact(d.instance, arena.allocateFrom(uri), out),
                    "IREE device create '" + uri + "'");
            d.device = out.get(ADDRESS, 0);
            return d;
        } catch (IreeRuntimeException e) {
            d.close();
            throw e;
        } catch (Throwable t) {
            d.close();
            throw new IreeRuntimeException("IREE device create '" + uri + "'", t);
        }
    }

    public String uri() {
        return uri;
    }

    /**
     * Loads a compiled {@code .vmfb} into a new session on this device. The bytes are copied into
     * native memory kept alive until the returned module is closed.
     */
    public synchronized IreeModule load(byte[] vmfb) {
        ensureOpen();
        Arena arena = Arena.ofShared();
        MemorySegment session = MemorySegment.NULL;
        try {
            MemorySegment data = arena.allocate(vmfb.length, 64);
            MemorySegment.copy(vmfb, 0, data, ValueLayout.JAVA_BYTE, 0, vmfb.length);
            MemorySegment out = arena.allocate(ADDRESS);
            lib.check(
                    (MemorySegment) lib.sessionCreate.invokeExact(instance, device, out),
                    "IREE session create");
            session = out.get(ADDRESS, 0);
            lib.check(
                    (MemorySegment)
                            lib.sessionAppendModule.invokeExact(session, data, (long) vmfb.length),
                    "IREE module load");
            return new IreeModule(this, session, arena);
        } catch (Throwable t) {
            if (!session.equals(MemorySegment.NULL)) {
                releaseSession(session);
            }
            arena.close();
            if (t instanceof IreeRuntimeException e) {
                throw e;
            }
            throw new IreeRuntimeException("IREE module load", t);
        }
    }

    /** Calls {@code function} (e.g. {@code module.main}) of a loaded session. */
    synchronized List<HostTensor> call(
            MemorySegment session, String function, List<HostTensor> inputs) {
        ensureOpen();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment out = arena.allocate(ADDRESS);
            lib.check(
                    (MemorySegment)
                            lib.callCreate.invokeExact(session, arena.allocateFrom(function), out),
                    "IREE call '" + function + "'");
            MemorySegment call = out.get(ADDRESS, 0);
            try {
                for (int i = 0; i < inputs.size(); i++) {
                    HostTensor in = inputs.get(i);
                    MemorySegment shape = arena.allocateFrom(JAVA_LONG, in.shape());
                    MemorySegment data = in.data();
                    if (!data.isNative()) {
                        data = arena.allocate(data.byteSize(), 64).copyFrom(data);
                    }
                    lib.check(
                            (MemorySegment)
                                    lib.callPushInput.invokeExact(
                                            call,
                                            shape,
                                            (long) in.shape().length,
                                            in.elementType(),
                                            data,
                                            data.byteSize()),
                            "IREE input " + i + " of '" + function + "'");
                }
                lib.check(
                        (MemorySegment) lib.callInvoke.invokeExact(call),
                        "IREE invoke '" + function + "'");
                long count = (long) lib.callOutputCount.invokeExact(call);
                var outputs = new ArrayList<HostTensor>((int) count);
                for (long k = 0; k < count; k++) {
                    lib.check(
                            (MemorySegment) lib.callPopOutput.invokeExact(call, out),
                            "IREE output " + k + " of '" + function + "'");
                    MemorySegment view = out.get(ADDRESS, 0);
                    try {
                        outputs.add(read(session, view));
                    } finally {
                        lib.bufferViewRelease.invokeExact(view);
                    }
                }
                return outputs;
            } finally {
                lib.callRelease.invokeExact(call);
            }
        } catch (IreeRuntimeException e) {
            throw e;
        } catch (Throwable t) {
            throw new IreeRuntimeException("IREE call '" + function + "'", t);
        }
    }

    private HostTensor read(MemorySegment session, MemorySegment view) throws Throwable {
        int rank = (int) (long) lib.bufferViewRank.invokeExact(view);
        long[] shape = new long[rank];
        for (int i = 0; i < rank; i++) {
            shape[i] = (long) lib.bufferViewDim.invokeExact(view, (long) i);
        }
        int elementType = (int) lib.bufferViewElementType.invokeExact(view);
        long bytes = (long) lib.bufferViewByteLength.invokeExact(view);
        MemorySegment data = Arena.ofAuto().allocate(Math.max(bytes, 1), 64).asSlice(0, bytes);
        lib.check(
                (MemorySegment) lib.bufferViewRead.invokeExact(session, view, data, bytes),
                "IREE output read");
        return new HostTensor(elementType, shape, data);
    }

    synchronized void releaseSession(MemorySegment session) {
        try {
            lib.sessionRelease.invokeExact(session);
        } catch (Throwable t) {
            throw new IreeRuntimeException("IREE session release", t);
        }
    }

    private void ensureOpen() {
        if (device == null) {
            throw new IllegalStateException("IREE device " + uri + " is closed");
        }
    }

    @Override
    public synchronized void close() {
        try {
            if (device != null && !device.equals(MemorySegment.NULL)) {
                lib.deviceRelease.invokeExact(device);
            }
            if (instance != null && !instance.equals(MemorySegment.NULL)) {
                lib.instanceRelease.invokeExact(instance);
            }
        } catch (Throwable t) {
            throw new IreeRuntimeException("IREE device close", t);
        } finally {
            device = null;
            instance = null;
        }
    }
}
