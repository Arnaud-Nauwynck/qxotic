package com.qxotic.jota.runtime.iree.compiler;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.util.Locale;
import java.util.Optional;

/**
 * Keeps the JVM's POSIX signal handlers in place around calls into the IREE compiler.
 *
 * <p>LLVM, inside {@code libIREECompiler}, lazily installs crash handlers for SIGSEGV, SIGBUS,
 * SIGILL, SIGFPE, ... (e.g. when the {@code llvm-cpu} backend writes temporary link files through
 * {@code RemoveFileOnSignal}). Those replace the JVM's handlers, and the JVM relies on SIGSEGV for
 * ordinary work (implicit null checks, safepoint polls): the next such fault reaches LLVM's
 * handler, which re-raises it, and the JVM dies in {@code tgkill} with a "crash" in libc.
 *
 * <p>The handlers are captured once, before the compiler library is loaded, and written back after
 * every compiler call. LLVM registers its handlers only once per process, so after the first
 * restore it does not take them over again. A JVM-internal fault on another thread <em>during</em>
 * the compile that first installs them can still hit LLVM's handler; preloading the JVM's
 * signal-chaining library ({@code LD_PRELOAD=$JAVA_HOME/lib/libjsig.so}) removes that window.
 *
 * <p>No-op on Windows.
 */
final class SignalHandlerGuard {

    /** Signals 1..31; sigaction() rejects SIGKILL / SIGSTOP, which is ignored. */
    private static final int MAX_SIGNAL = 31;

    /** Larger than {@code struct sigaction} on every supported libc (152 bytes on glibc x86_64). */
    private static final long SIGACTION_BYTES = 256;

    private final MethodHandle sigaction;
    private final MemorySegment saved;
    private final boolean[] captured;

    private SignalHandlerGuard(MethodHandle sigaction, MemorySegment saved, boolean[] captured) {
        this.sigaction = sigaction;
        this.saved = saved;
        this.captured = captured;
    }

    /** Captures the current handlers, or empty where there is no {@code sigaction}. */
    static Optional<SignalHandlerGuard> capture() {
        if (System.getProperty("os.name", "").toLowerCase(Locale.ROOT).contains("win")) {
            return Optional.empty();
        }
        Linker linker = Linker.nativeLinker();
        Optional<MemorySegment> symbol = linker.defaultLookup().find("sigaction");
        if (symbol.isEmpty()) {
            return Optional.empty();
        }
        MethodHandle sigaction =
                linker.downcallHandle(
                        symbol.get(), FunctionDescriptor.of(JAVA_INT, JAVA_INT, ADDRESS, ADDRESS));
        MemorySegment saved = Arena.global().allocate(SIGACTION_BYTES * (MAX_SIGNAL + 1), 16);
        boolean[] captured = new boolean[MAX_SIGNAL + 1];
        for (int sig = 1; sig <= MAX_SIGNAL; sig++) {
            try {
                int rc = (int) sigaction.invokeExact(sig, MemorySegment.NULL, slot(saved, sig));
                captured[sig] = rc == 0;
            } catch (Throwable t) {
                return Optional.empty();
            }
        }
        return Optional.of(new SignalHandlerGuard(sigaction, saved, captured));
    }

    /** Writes back every captured handler that has been replaced since. */
    void restore() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment current = arena.allocate(SIGACTION_BYTES, 16);
            for (int sig = 1; sig <= MAX_SIGNAL; sig++) {
                if (!captured[sig]) {
                    continue; // e.g. SIGKILL, SIGSTOP
                }
                MemorySegment slot = slot(saved, sig);
                long savedHandler = slot.get(JAVA_LONG, 0);
                int rc = (int) sigaction.invokeExact(sig, MemorySegment.NULL, current);
                if (rc == 0 && current.get(JAVA_LONG, 0) != savedHandler) {
                    int ignored = (int) sigaction.invokeExact(sig, slot, MemorySegment.NULL);
                }
            }
        } catch (Throwable t) {
            throw new IreeCompilerException("Cannot restore JVM signal handlers", t);
        }
    }

    /** The handler address currently installed for {@code sig} (0 for SIG_DFL), for tests. */
    static long currentHandler(int sig) {
        Linker linker = Linker.nativeLinker();
        MethodHandle sigaction =
                linker.downcallHandle(
                        linker.defaultLookup().find("sigaction").orElseThrow(),
                        FunctionDescriptor.of(JAVA_INT, JAVA_INT, ADDRESS, ADDRESS));
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment current = arena.allocate(SIGACTION_BYTES, 16);
            int rc = (int) sigaction.invokeExact(sig, MemorySegment.NULL, current);
            if (rc != 0) {
                throw new IllegalStateException("sigaction(" + sig + ") failed");
            }
            return current.get(JAVA_LONG, 0);
        } catch (RuntimeException e) {
            throw e;
        } catch (Throwable t) {
            throw new IllegalStateException(t);
        }
    }

    private static MemorySegment slot(MemorySegment all, int sig) {
        return all.asSlice(sig * SIGACTION_BYTES, SIGACTION_BYTES);
    }
}
