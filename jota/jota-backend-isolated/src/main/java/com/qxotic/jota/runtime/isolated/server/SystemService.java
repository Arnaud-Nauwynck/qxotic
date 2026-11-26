package com.qxotic.jota.runtime.isolated.server;

import com.qxotic.jota.runtime.isolated.spi.CallContext;
import com.qxotic.jota.runtime.isolated.spi.IsolatedService;
import com.qxotic.jota.runtime.isolated.wire.Wire;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.concurrent.CancellationException;

/**
 * Built-in service {@code "system"}, always present in the child: liveness, introspection and, only
 * when the child runs with {@code -Djota.isolated.testCrashes=true}, deliberate crashes to test the
 * parent's recovery.
 */
public final class SystemService implements IsolatedService {

    public static final String NAME = "system";
    public static final String TEST_CRASHES_PROPERTY = "jota.isolated.testCrashes";

    @Override
    public String name() {
        return NAME;
    }

    @Override
    public byte[] invoke(String method, byte[] payload, CallContext context) throws Exception {
        return switch (method) {
            case "ping" -> new byte[0];
            case "echo" -> payload;
            case "pid" -> Wire.writer().i64(ProcessHandle.current().pid()).toByteArray();
            case "property" -> {
                String value = System.getProperty(Wire.reader(payload).string());
                yield Wire.writer().string(value).toByteArray();
            }
            case "sleep" -> {
                long millis = Wire.reader(payload).i64();
                long deadline = System.nanoTime() + millis * 1_000_000L;
                while (System.nanoTime() < deadline) {
                    if (context.isCancelled()) {
                        throw new CancellationException("sleep cancelled");
                    }
                    Thread.sleep(Math.min(10, Math.max(1, millis)));
                }
                yield new byte[0];
            }
            case "stream" -> {
                int count = Wire.reader(payload).i32();
                for (int i = 0; i < count && !context.isCancelled(); i++) {
                    context.emit(Wire.writer().i32(i).toByteArray());
                }
                yield Wire.writer().i32(count).toByteArray();
            }
            case "fail" -> throw new IllegalStateException(Wire.reader(payload).string());
            case "halt" -> {
                requireTestCrashes();
                Runtime.getRuntime().halt(Wire.reader(payload).i32());
                yield new byte[0];
            }
            case "segv" -> {
                requireTestCrashes();
                // A real native fault: a write at address 0, as a buggy native library would do.
                MemorySegment.NULL.reinterpret(8).set(ValueLayout.JAVA_LONG, 0, 42L);
                yield new byte[0];
            }
            case "abort" -> {
                requireTestCrashes();
                // libc abort(): raises SIGABRT, which HotSpot does not handle (no hs_err), as an
                // assert() failing in a native library would.
                Linker linker = Linker.nativeLinker();
                MemorySegment abort =
                        linker.defaultLookup()
                                .find("abort")
                                .orElseThrow(() -> new UnsupportedOperationException("no abort"));
                try {
                    linker.downcallHandle(abort, FunctionDescriptor.ofVoid()).invokeExact();
                } catch (Throwable t) {
                    throw new IllegalStateException("abort() returned", t);
                }
                yield new byte[0];
            }
            default -> throw new UnsupportedOperationException("system." + method);
        };
    }

    private static void requireTestCrashes() {
        if (!Boolean.getBoolean(TEST_CRASHES_PROPERTY)) {
            throw new SecurityException(
                    "Crash methods need -D" + TEST_CRASHES_PROPERTY + "=true in the child");
        }
    }
}
