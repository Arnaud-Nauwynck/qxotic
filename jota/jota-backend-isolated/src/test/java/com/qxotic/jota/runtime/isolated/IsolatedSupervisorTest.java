package com.qxotic.jota.runtime.isolated;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jota.runtime.isolated.wire.Wire;
import java.time.Duration;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

@Timeout(value = 90, unit = TimeUnit.SECONDS)
class IsolatedSupervisorTest {

    private static void crash(IsolatedJvm jvm) {
        assertThrows(
                IsolatedCrashException.class,
                () -> jvm.invoke("system", "halt", Wire.writer().i32(9).toByteArray()));
    }

    @Test
    void restartsAfterACrashAndReplaysStartHooks() {
        try (IsolatedSupervisor supervisor =
                new IsolatedSupervisor(
                        IsolatedJvmTest.testConfig().build(), 3, Duration.ofMinutes(1))) {
            AtomicInteger hookRuns = new AtomicInteger();
            // a "recipe": state every new child must have, e.g. a loaded model
            supervisor.onStart(
                    jvm -> {
                        hookRuns.incrementAndGet();
                        jvm.invoke("test", "increment", null);
                    });
            IsolatedJvm first = supervisor.get();
            assertEquals(1, hookRuns.get());
            assertEquals(2, Wire.reader(supervisor.invoke("test", "increment", null)).i64());

            crash(first);

            IsolatedJvm second = supervisor.get();
            assertNotEquals(first.pid(), second.pid());
            assertEquals(2, hookRuns.get());
            assertEquals(2, supervisor.starts());
            // state was replayed by the hook, not carried over
            assertEquals(2, Wire.reader(supervisor.invoke("test", "increment", null)).i64());
        }
    }

    @Test
    void givesUpAfterTooManyCrashes() {
        try (IsolatedSupervisor supervisor =
                new IsolatedSupervisor(
                        IsolatedJvmTest.testConfig().build(), 1, Duration.ofMinutes(1))) {
            crash(supervisor.get());
            crash(supervisor.get()); // one restart allowed
            IsolatedException e = assertThrows(IsolatedException.class, supervisor::get);
            assertEquals(IsolatedException.class, e.getClass());
            assertThrows(IsolatedException.class, supervisor::get); // stays open
        }
    }
}
