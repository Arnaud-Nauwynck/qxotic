package com.qxotic.jota.runtime.isolated;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.runtime.isolated.server.SystemService;
import com.qxotic.jota.runtime.isolated.wire.Wire;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.lang.foreign.ValueLayout;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.condition.EnabledOnOs;
import org.junit.jupiter.api.condition.OS;
import org.junit.jupiter.api.io.TempDir;

@Timeout(value = 60, unit = TimeUnit.SECONDS)
class IsolatedJvmTest {

    /** Shared by the tests that leave the child alive: a JVM start costs ~0.3 s. */
    private static IsolatedJvm shared;

    static IsolationConfig.Builder testConfig() {
        return IsolationConfig.builder()
                .systemProperty(SystemService.TEST_CRASHES_PROPERTY, "true")
                .addJvmOption("-XX:-CreateCoredumpOnCrash")
                .serverThreads(4);
    }

    @BeforeAll
    static void startShared() {
        shared = IsolatedJvm.start(testConfig().build());
    }

    @AfterAll
    static void closeShared() {
        if (shared != null) {
            shared.close();
        }
    }

    @Test
    void childIsAnotherProcess() {
        long childPid = Wire.reader(shared.invoke("system", "pid", null)).i64();
        assertEquals(shared.pid(), childPid);
        assertNotEquals(ProcessHandle.current().pid(), childPid);
        assertTrue(shared.isAlive());
    }

    @Test
    void echoRoundTrip() {
        byte[] big = new byte[3 << 20];
        for (int i = 0; i < big.length; i++) {
            big[i] = (byte) (i * 31);
        }
        assertArrayEquals(big, shared.invoke("system", "echo", big));
        assertEquals(0, shared.invoke("system", "ping", null).length);
    }

    @Test
    void systemPropertiesReachTheChild() {
        byte[] name = Wire.writer().string(SystemService.TEST_CRASHES_PROPERTY).toByteArray();
        assertEquals("true", Wire.reader(shared.invoke("system", "property", name)).string());
    }

    @Test
    void servicesAreFoundWithServiceLoaderAndKeepState() {
        long a = Wire.reader(shared.invoke("test", "increment", null)).i64();
        long b = Wire.reader(shared.invoke("test", "increment", null)).i64();
        assertEquals(a + 1, b);
    }

    @Test
    void remoteExceptionsLeaveTheChildAlive() {
        byte[] msg = Wire.writer().string("boom").toByteArray();
        IsolatedRemoteException e =
                assertThrows(
                        IsolatedRemoteException.class, () -> shared.invoke("system", "fail", msg));
        assertEquals(IllegalStateException.class.getName(), e.remoteClass());
        assertEquals("boom", e.remoteMessage());
        assertTrue(e.remoteStackTrace().contains("SystemService"));

        IsolatedRemoteException unknown =
                assertThrows(IsolatedRemoteException.class, () -> shared.invoke("nope", "x", null));
        assertTrue(unknown.getMessage().contains("No service 'nope'"));
        assertTrue(shared.isAlive());
    }

    @Test
    void streamsItemsInOrderBeforeTheResponse() throws Exception {
        List<Integer> items = new CopyOnWriteArrayList<>();
        byte[] result =
                shared.callStreaming(
                                "system",
                                "stream",
                                Wire.writer().i32(1000).toByteArray(),
                                item -> items.add(Wire.reader(item).i32()))
                        .get(10, TimeUnit.SECONDS);
        assertEquals(1000, Wire.reader(result).i32());
        assertEquals(1000, items.size());
        for (int i = 0; i < items.size(); i++) {
            assertEquals(i, items.get(i));
        }
    }

    @Test
    void concurrentCallsAreMultiplexed() throws Exception {
        List<CompletableFuture<byte[]>> futures = new java.util.ArrayList<>();
        for (int i = 0; i < 200; i++) {
            futures.add(shared.call("system", "echo", Wire.writer().i32(i).toByteArray()));
        }
        for (int i = 0; i < futures.size(); i++) {
            assertEquals(i, Wire.reader(futures.get(i).get(10, TimeUnit.SECONDS)).i32());
        }
    }

    @Test
    void timeoutCancelsTheCallInTheChild() {
        byte[] tenSeconds = Wire.writer().i64(10_000).toByteArray();
        long start = System.nanoTime();
        assertThrows(
                IsolatedTimeoutException.class,
                () -> shared.invoke("system", "sleep", tenSeconds, Duration.ofMillis(200)));
        // the worker noticed the cancel flag: the next call is served at once
        shared.invoke("system", "ping", null, Duration.ofSeconds(5));
        assertTrue(System.nanoTime() - start < TimeUnit.SECONDS.toNanos(5));
    }

    @Test
    void sharedMemoryIsSeenByBothSides() {
        int n = 1 << 20;
        try (SharedMemoryFile shm = SharedMemoryFile.create(n * 4L)) {
            for (int i = 0; i < n; i++) {
                shm.segment().setAtIndex(ValueLayout.JAVA_FLOAT, i, 1f);
            }
            byte[] request =
                    Wire.writer().string(shm.path().toString()).i32(n).f64(3).toByteArray();
            double sum = Wire.reader(shared.invoke("test", "scaleShm", request)).f64();
            assertEquals(3.0 * n, sum);
            assertEquals(3f, shm.segment().getAtIndex(ValueLayout.JAVA_FLOAT, n - 1));
        }
    }

    @Test
    void haltFailsPendingAndLaterCalls(@TempDir Path dir) throws Exception {
        IsolatedJvm jvm = IsolatedJvm.start(testConfig().workDirectory(dir).build());
        CompletableFuture<byte[]> slow =
                jvm.call("system", "sleep", Wire.writer().i64(30_000).toByteArray());
        CompletableFuture<byte[]> halt =
                jvm.call("system", "halt", Wire.writer().i32(77).toByteArray());

        ExecutionException e =
                assertThrows(ExecutionException.class, () -> halt.get(10, TimeUnit.SECONDS));
        IsolatedCrashException crash = assertInstanceOf(IsolatedCrashException.class, e.getCause());
        assertEquals(77, crash.report().exitCode());
        assertThrows(ExecutionException.class, () -> slow.get(10, TimeUnit.SECONDS));
        assertFalse(jvm.isAlive());
        assertThrows(IsolatedCrashException.class, () -> jvm.invoke("system", "ping", null));
        jvm.close();
        assertTrue(shared.isAlive(), "other children are unaffected");
    }

    @Test
    @EnabledOnOs({OS.LINUX, OS.MAC})
    void nativeSegfaultIsReportedNotFatal(@TempDir Path dir) {
        IsolatedJvm jvm = IsolatedJvm.start(testConfig().workDirectory(dir).build());
        IsolatedCrashException crash =
                assertThrows(
                        IsolatedCrashException.class,
                        () -> jvm.invoke("system", "segv", null, Duration.ofSeconds(30)));
        // HotSpot's own handler catches the fault and writes hs_err; with core dumps off it then
        // exits with 1 instead of abort()ing, so the cause comes from hs_err, not the exit code.
        assertNotEquals(0, crash.report().exitCode());
        Path hsErr = crash.report().hsErrFile().orElseThrow();
        String fatal = crash.report().fatalError().orElseThrow();
        assertTrue(fatal.contains("SIGSEGV"), fatal);
        assertTrue(crash.getMessage().contains("SIGSEGV"), crash.getMessage());
        assertTrue(Files.exists(hsErr));
        jvm.close();
        // and this JVM is fine
        assertTrue(shared.isAlive());
        assertEquals(0, shared.invoke("system", "ping", null).length);
    }

    @Test
    void crashMethodsAreOffByDefault(@TempDir Path dir) {
        try (IsolatedJvm jvm =
                IsolatedJvm.start(IsolationConfig.builder().workDirectory(dir).build())) {
            IsolatedRemoteException e =
                    assertThrows(
                            IsolatedRemoteException.class,
                            () -> jvm.invoke("system", "segv", null));
            assertEquals(SecurityException.class.getName(), e.remoteClass());
            assertTrue(jvm.isAlive());
        }
    }

    @Test
    void killedChildIsReported(@TempDir Path dir) {
        IsolatedJvm jvm = IsolatedJvm.start(testConfig().workDirectory(dir).build());
        CompletableFuture<byte[]> slow =
                jvm.call("system", "sleep", Wire.writer().i64(30_000).toByteArray());
        jvm.kill();
        ExecutionException e =
                assertThrows(ExecutionException.class, () -> slow.get(10, TimeUnit.SECONDS));
        IsolatedCrashException crash = assertInstanceOf(IsolatedCrashException.class, e.getCause());
        if (!IsolationConfig.isWindows()) {
            assertEquals("SIGKILL", crash.report().signal().orElseThrow());
        }
        jvm.close();
    }

    @Test
    void closeStopsTheChild() throws Exception {
        IsolatedJvm jvm = IsolatedJvm.start(testConfig().build());
        Path dir = jvm.workDirectory();
        ProcessHandle child = ProcessHandle.of(jvm.pid()).orElseThrow();
        jvm.close();
        child.onExit().get(10, TimeUnit.SECONDS);
        assertFalse(child.isAlive());
        assertFalse(Files.exists(dir), "an owned work directory is deleted on a clean close");
        IsolatedException e =
                assertThrows(IsolatedException.class, () -> jvm.invoke("system", "ping", null));
        assertFalse(e instanceof IsolatedCrashException);
        jvm.close(); // idempotent
    }

    @Test
    void childThatCannotStartFailsFast(@TempDir Path dir) {
        IsolationConfig bad =
                testConfig().workDirectory(dir).mainClass("no.such.MainClass").build();
        long start = System.nanoTime();
        IsolatedCrashException e =
                assertThrows(IsolatedCrashException.class, () -> IsolatedJvm.start(bad));
        assertNotEquals(0, e.report().exitCode());
        assertTrue(System.nanoTime() - start < TimeUnit.SECONDS.toNanos(20));
    }

    @Test
    @EnabledOnOs({OS.LINUX, OS.MAC})
    void nativeAbortIsReported(@TempDir Path dir) {
        IsolatedJvm jvm = IsolatedJvm.start(testConfig().workDirectory(dir).build());
        IsolatedCrashException crash =
                assertThrows(
                        IsolatedCrashException.class,
                        () -> jvm.invoke("system", "abort", null, Duration.ofSeconds(30)));
        assertEquals("SIGABRT", crash.report().signal().orElseThrow(), crash.getMessage());
        jvm.close();
        assertTrue(shared.isAlive());
    }

    @Test
    void timeoutCanKillAHungChild(@TempDir Path dir) throws Exception {
        IsolatedJvm jvm =
                IsolatedJvm.start(
                        testConfig()
                                .workDirectory(dir)
                                .timeoutAction(IsolationConfig.TimeoutAction.KILL)
                                .build());
        CompletableFuture<byte[]> other =
                jvm.call("system", "sleep", Wire.writer().i64(30_000).toByteArray());
        byte[] tenSeconds = Wire.writer().i64(10_000).toByteArray();
        IsolatedTimeoutException timeout =
                assertThrows(
                        IsolatedTimeoutException.class,
                        () -> jvm.invoke("system", "sleep", tenSeconds, Duration.ofMillis(300)));
        assertTrue(timeout.getMessage().contains("killed"), timeout.getMessage());
        ExecutionException e =
                assertThrows(ExecutionException.class, () -> other.get(10, TimeUnit.SECONDS));
        IsolatedCrashException crash = assertInstanceOf(IsolatedCrashException.class, e.getCause());
        assertTrue(
                crash.report().reason().contains("system.sleep did not answer"),
                crash.getMessage());
        assertFalse(jvm.isAlive());
        jvm.close();
    }

    @Test
    void heartbeatDoesNotTripDuringLongCalls(@TempDir Path dir) {
        try (IsolatedJvm jvm =
                IsolatedJvm.start(
                        testConfig()
                                .workDirectory(dir)
                                .serverThreads(1)
                                .heartbeat(Duration.ofMillis(100), Duration.ofMillis(800))
                                .build())) {
            // the only worker is busy for 2 s, far longer than the heartbeat timeout
            jvm.invoke("system", "sleep", Wire.writer().i64(2_000).toByteArray());
            assertTrue(jvm.isAlive());
        }
    }

    @Test
    @EnabledOnOs({OS.LINUX, OS.MAC})
    void heartbeatKillsAWedgedChild(@TempDir Path dir) throws Exception {
        IsolatedJvm jvm =
                IsolatedJvm.start(
                        testConfig()
                                .workDirectory(dir)
                                .heartbeat(Duration.ofMillis(100), Duration.ofMillis(800))
                                .build());
        CompletableFuture<byte[]> pending =
                jvm.call("system", "sleep", Wire.writer().i64(30_000).toByteArray());
        // SIGSTOP: the process is alive but answers nothing, as a wedged JVM
        new ProcessBuilder("kill", "-STOP", Long.toString(jvm.pid())).start().waitFor();
        ExecutionException e =
                assertThrows(ExecutionException.class, () -> pending.get(20, TimeUnit.SECONDS));
        IsolatedCrashException crash = assertInstanceOf(IsolatedCrashException.class, e.getCause());
        assertTrue(crash.report().reason().contains("no heartbeat"), crash.getMessage());
        jvm.close();
    }

    @Test
    void childDiesWithItsParent(@TempDir Path dir) throws Exception {
        Path java = IsolationConfig.defaults().javaExecutable();
        Process middle =
                new ProcessBuilder(
                                java.toString(),
                                "--enable-native-access=ALL-UNNAMED",
                                "-cp",
                                System.getProperty("java.class.path"),
                                MiddleJvmMain.class.getName(),
                                dir.toString())
                        .redirectError(ProcessBuilder.Redirect.INHERIT)
                        .start();
        try {
            BufferedReader out =
                    new BufferedReader(
                            new InputStreamReader(middle.getInputStream(), StandardCharsets.UTF_8));
            String line = out.readLine();
            assertTrue(line != null && line.startsWith("child-pid "), String.valueOf(line));
            ProcessHandle child =
                    ProcessHandle.of(Long.parseLong(line.substring("child-pid ".length())))
                            .orElseThrow();
            assertTrue(child.isAlive());
            middle.destroyForcibly().waitFor(10, TimeUnit.SECONDS); // SIGKILL: no hook runs
            child.onExit().get(10, TimeUnit.SECONDS);
            assertFalse(child.isAlive());
        } finally {
            middle.destroyForcibly();
        }
    }
}
