package com.qxotic.jota.runtime.isolated;

import java.time.Duration;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.function.Consumer;

/**
 * Keeps one child JVM available: starts it lazily, restarts it after a crash, and gives up (circuit
 * open) after {@code maxRestarts} crashes within {@code window}.
 *
 * <p>State that lived in the dead child (a loaded model, registered kernels) is gone; clients
 * register {@link #onStart} hooks that replay it in each new child, before any call reaches it.
 * Calls in flight during a crash still fail: only new calls see the new child.
 */
public final class IsolatedSupervisor implements AutoCloseable {

    private final IsolationConfig config;
    private final int maxRestarts;
    private final Duration window;
    private final List<Consumer<IsolatedJvm>> startHooks = new CopyOnWriteArrayList<>();
    private final Deque<Long> crashTimes = new ArrayDeque<>();
    private IsolatedJvm current;
    private int starts;
    private boolean closed;
    private IsolatedException lastCrash;

    public IsolatedSupervisor(IsolationConfig config, int maxRestarts, Duration window) {
        this.config = Objects.requireNonNull(config, "config");
        if (maxRestarts < 0) {
            throw new IllegalArgumentException("maxRestarts < 0");
        }
        this.maxRestarts = maxRestarts;
        this.window = Objects.requireNonNull(window, "window");
    }

    /** Runs {@code hook} on every child this supervisor starts, including the current one. */
    public synchronized void onStart(Consumer<IsolatedJvm> hook) {
        startHooks.add(Objects.requireNonNull(hook, "hook"));
        if (current != null && current.isAlive()) {
            hook.accept(current);
        }
    }

    /** The live child, started or restarted if needed. */
    public synchronized IsolatedJvm get() {
        if (closed) {
            throw new IsolatedException("Supervisor closed");
        }
        if (current != null && current.isAlive()) {
            return current;
        }
        if (current != null) {
            recordCrash();
        }
        current = IsolatedJvm.start(config);
        starts++;
        IsolatedJvm jvm = current;
        jvm.terminated()
                .thenAccept(
                        cause -> {
                            synchronized (this) {
                                if (cause instanceof IsolatedCrashException) {
                                    lastCrash = cause;
                                }
                            }
                        });
        try {
            for (Consumer<IsolatedJvm> hook : startHooks) {
                hook.accept(jvm);
            }
        } catch (RuntimeException e) {
            jvm.close();
            throw new IsolatedException("A start hook failed in the new isolated JVM", e);
        }
        return jvm;
    }

    /** Convenience: a blocking call on the live child. */
    public byte[] invoke(String service, String method, byte[] payload) {
        return get().invoke(service, method, payload);
    }

    private void recordCrash() {
        long now = System.nanoTime();
        crashTimes.addLast(now);
        while (!crashTimes.isEmpty() && now - crashTimes.peekFirst() > window.toNanos()) {
            crashTimes.removeFirst();
        }
        if (crashTimes.size() > maxRestarts) {
            closed = true;
            throw new IsolatedException(
                    "The isolated JVM crashed "
                            + crashTimes.size()
                            + " times within "
                            + window
                            + "; giving up",
                    lastCrash);
        }
    }

    /** How many children were started so far. */
    public synchronized int starts() {
        return starts;
    }

    /** The exception the last crashed child failed its calls with, or null. */
    public synchronized IsolatedException lastCrash() {
        return lastCrash;
    }

    @Override
    public synchronized void close() {
        closed = true;
        if (current != null) {
            current.close();
            current = null;
        }
    }
}
