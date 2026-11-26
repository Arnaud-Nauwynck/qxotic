package com.qxotic.jota.runtime.isolated;

import com.qxotic.jota.runtime.isolated.IsolatedCrashException.CrashReport;
import com.qxotic.jota.runtime.isolated.server.IsolatedServerMain;
import com.qxotic.jota.runtime.isolated.wire.Frame;
import com.qxotic.jota.runtime.isolated.wire.FrameIO;
import com.qxotic.jota.runtime.isolated.wire.FrameType;
import com.qxotic.jota.runtime.isolated.wire.Wire;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.io.UncheckedIOException;
import java.net.StandardProtocolFamily;
import java.net.UnixDomainSocketAddress;
import java.nio.channels.ServerSocketChannel;
import java.nio.channels.SocketChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.PosixFilePermissions;
import java.security.SecureRandom;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;
import java.util.stream.Stream;

/**
 * One child JVM and the connection to it.
 *
 * <p>Calls name a service hosted in the child and a method; they return futures. When the child
 * dies, every pending and later call fails with {@link IsolatedCrashException}; this JVM is not
 * affected. An {@code IsolatedJvm} is never restarted: start a new one, or use {@link
 * IsolatedSupervisor}.
 *
 * <p>Threads: one reader and one writer per child, plus one heartbeat thread when configured.
 * Stream callbacks run on the reader thread and must return quickly.
 */
public final class IsolatedJvm implements AutoCloseable {

    private static final String SOCKET_NAME = "s.sock";
    private static final int STDERR_TAIL_BYTES = 8192;

    private final IsolationConfig config;
    private final Path workDir;
    private final boolean ownsWorkDir;
    private final Process process;
    private final SocketChannel channel;
    private final FrameIO io;
    private final Map<Long, Pending> pending = new ConcurrentHashMap<>();
    private final AtomicLong nextCallId = new AtomicLong(1);
    private final LinkedBlockingQueue<Frame> outbox = new LinkedBlockingQueue<>();
    private final AtomicReference<IsolatedException> death = new AtomicReference<>();
    private final CompletableFuture<IsolatedException> terminated = new CompletableFuture<>();
    private final Thread reader;
    private final Thread writer;
    private volatile boolean closing;

    /** Why this side killed the child, for the crash report; null if it died on its own. */
    private volatile String killReason;

    private volatile long lastPongNanos;

    private record Pending(CompletableFuture<byte[]> future, Consumer<byte[]> onItem) {}

    private IsolatedJvm(
            IsolationConfig config,
            Path workDir,
            boolean ownsWorkDir,
            Process process,
            SocketChannel channel) {
        this.config = config;
        this.workDir = workDir;
        this.ownsWorkDir = ownsWorkDir;
        this.process = process;
        this.channel = channel;
        this.io = new FrameIO(config.maxFrameBytes());
        this.reader =
                Thread.ofPlatform()
                        .daemon()
                        .name("isolated-reader-" + pid())
                        .unstarted(this::readLoop);
        this.writer =
                Thread.ofPlatform()
                        .daemon()
                        .name("isolated-writer-" + pid())
                        .unstarted(this::writeLoop);
    }

    // ------------------------------------------------------------------ start

    /** Launches a child JVM with {@code config} and waits until it is connected and ready. */
    public static IsolatedJvm start(IsolationConfig config) {
        Objects.requireNonNull(config, "config");
        Path workDir;
        boolean ownsWorkDir = config.workDirectory() == null;
        try {
            workDir = ownsWorkDir ? createPrivateTempDir() : config.workDirectory();
            Files.createDirectories(workDir);
        } catch (IOException e) {
            throw new IsolatedException("Cannot create the work directory", e);
        }
        Path socketPath = workDir.resolve(SOCKET_NAME);
        String token = newToken();
        Process process = null;
        try (ServerSocketChannel server = ServerSocketChannel.open(StandardProtocolFamily.UNIX)) {
            Files.deleteIfExists(socketPath);
            server.bind(UnixDomainSocketAddress.of(socketPath));

            List<String> command = new ArrayList<>(config.command());
            command.add(command.size() - 1, "-XX:ErrorFile=" + workDir.resolve("hs_err_pid%p.log"));
            command.add(socketPath.toString());
            ProcessBuilder pb = new ProcessBuilder(command).directory(workDir.toFile());
            pb.environment().putAll(config.environment());
            pb.environment().put(IsolatedServerMain.TOKEN_ENV, token);
            if (config.inheritOutput()) {
                pb.inheritIO();
            } else {
                pb.redirectInput(ProcessBuilder.Redirect.PIPE);
                pb.redirectOutput(workDir.resolve("child.out").toFile());
                pb.redirectError(workDir.resolve("child.err").toFile());
            }
            process = pb.start();
            process.getOutputStream().close(); // the child reads nothing from stdin

            SocketChannel channel = accept(server, process, config.startupTimeout());
            Files.deleteIfExists(socketPath); // connected: nobody else may connect
            IsolatedJvm jvm = new IsolatedJvm(config, workDir, ownsWorkDir, process, channel);
            jvm.handshake(token, config.startupTimeout());
            jvm.startHeartbeat();
            return jvm;
        } catch (IOException e) {
            if (process != null) {
                process.destroyForcibly();
            }
            throw new IsolatedException("Cannot start the isolated JVM in " + workDir, e);
        } catch (IsolatedException e) {
            if (process != null) {
                process.destroyForcibly();
            }
            throw e;
        }
    }

    private static SocketChannel accept(
            ServerSocketChannel server, Process process, Duration timeout) throws IOException {
        CompletableFuture<SocketChannel> accepted = new CompletableFuture<>();
        Thread acceptor =
                Thread.ofPlatform()
                        .daemon()
                        .name("isolated-accept")
                        .start(
                                () -> {
                                    try {
                                        accepted.complete(server.accept());
                                    } catch (IOException e) {
                                        accepted.completeExceptionally(e);
                                    }
                                });
        CompletableFuture<Object> first = CompletableFuture.anyOf(accepted, process.onExit());
        try {
            first.get(timeout.toMillis(), TimeUnit.MILLISECONDS);
        } catch (TimeoutException e) {
            server.close(); // unblocks the acceptor
            throw new IsolatedException(
                    "The isolated JVM did not connect within "
                            + timeout
                            + " (pid "
                            + process.pid()
                            + ")");
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            server.close();
            throw new IsolatedException("Interrupted while starting the isolated JVM", e);
        } catch (ExecutionException e) {
            throw new IOException(e.getCause());
        }
        if (!accepted.isDone()) {
            server.close();
            acceptor.interrupt();
            throw new IsolatedCrashException(
                    new CrashReport(
                            process.pid(),
                            process.exitValue(),
                            null,
                            Optional.empty(),
                            "exited before connecting"));
        }
        return accepted.join();
    }

    private void handshake(String token, Duration timeout) {
        CompletableFuture<byte[]> hello = new CompletableFuture<>();
        pending.put(0L, new Pending(hello, null));
        process.onExit().thenRun(this::onProcessExit);
        reader.start();
        writer.start();
        byte[] payload;
        try {
            payload = hello.get(timeout.toMillis(), TimeUnit.MILLISECONDS);
        } catch (Exception e) {
            kill();
            Throwable cause = e instanceof ExecutionException ? e.getCause() : e;
            if (cause instanceof IsolatedException ie) {
                throw ie;
            }
            throw new IsolatedException("No HELLO from the isolated JVM", cause);
        }
        Wire.Reader r = Wire.reader(payload);
        int version = r.i32();
        long childPid = r.i64();
        String childToken = r.string();
        if (version != FrameIO.PROTOCOL_VERSION) {
            kill();
            throw new IsolatedException(
                    "Protocol version mismatch: child "
                            + version
                            + ", parent "
                            + FrameIO.PROTOCOL_VERSION);
        }
        if (!token.equals(childToken) || childPid != process.pid()) {
            kill();
            throw new IsolatedException(
                    "The process that connected is not the child that was started");
        }
    }

    private void startHeartbeat() {
        lastPongNanos = System.nanoTime();
        if (config.heartbeatInterval() == null) {
            return;
        }
        Thread.ofPlatform().daemon().name("isolated-heartbeat-" + pid()).start(this::heartbeatLoop);
    }

    private void heartbeatLoop() {
        long interval = config.heartbeatInterval().toMillis();
        long timeout = config.heartbeatTimeout().toNanos();
        long seq = 0;
        try {
            while (death.get() == null && !closing) {
                outbox.add(Frame.of(FrameType.PING, ++seq));
                Thread.sleep(interval);
                long silent = System.nanoTime() - lastPongNanos;
                if (silent > timeout && death.get() == null && !closing) {
                    kill(
                            "no heartbeat answer for "
                                    + TimeUnit.NANOSECONDS.toMillis(silent)
                                    + " ms (wedged)");
                    return;
                }
            }
        } catch (InterruptedException e) {
            // stop
        }
    }

    // ------------------------------------------------------------------ calls

    /**
     * Starts a call; the future completes with the response bytes or an {@link IsolatedException}.
     */
    public CompletableFuture<byte[]> call(String service, String method, byte[] payload) {
        return callStreaming(service, method, payload, null);
    }

    /**
     * Starts a call whose service streams items (see {@code CallContext.emit}); {@code onItem} gets
     * them in order on the reader thread, before the future completes. Cancelling the future sends
     * a cancellation to the child.
     */
    public CompletableFuture<byte[]> callStreaming(
            String service, String method, byte[] payload, Consumer<byte[]> onItem) {
        Objects.requireNonNull(service, "service");
        Objects.requireNonNull(method, "method");
        long callId = nextCallId.getAndIncrement();
        CompletableFuture<byte[]> future = new CompletableFuture<>();
        IsolatedException dead = death.get();
        if (dead != null) {
            future.completeExceptionally(dead);
            return future;
        }
        byte[] request =
                Wire.writer()
                        .string(service)
                        .string(method)
                        .bytes(payload == null ? new byte[0] : payload)
                        .toByteArray();
        if (request.length + 9L > config.maxFrameBytes()) {
            future.completeExceptionally(
                    new IsolatedException(
                            "Request too large: "
                                    + request.length
                                    + " bytes, limit "
                                    + config.maxFrameBytes()
                                    + " (pass bulk data in shared memory)"));
            return future;
        }
        pending.put(callId, new Pending(future, onItem));
        future.whenComplete(
                (r, t) -> {
                    if (pending.remove(callId) != null && t instanceof CancellationException) {
                        outbox.add(Frame.of(FrameType.CANCEL, callId)); // caller gave up
                    }
                });
        // the process may have died between the check and the put
        dead = death.get();
        if (dead != null) {
            future.completeExceptionally(dead);
            return future;
        }
        outbox.add(new Frame(FrameType.REQUEST, callId, request));
        return future;
    }

    /**
     * A blocking call with the configured default timeout; unwraps the failure to an {@link
     * IsolatedException}.
     */
    public byte[] invoke(String service, String method, byte[] payload) {
        return invoke(service, method, payload, config.callTimeout());
    }

    /**
     * A blocking call; {@code timeout == null} waits forever. On timeout the call is cancelled in
     * the child, or the child is killed if the config says {@link
     * IsolationConfig.TimeoutAction#KILL}.
     */
    public byte[] invoke(String service, String method, byte[] payload, Duration timeout) {
        return await(call(service, method, payload), timeout, service + "." + method);
    }

    private byte[] await(CompletableFuture<byte[]> future, Duration timeout, String what) {
        try {
            return timeout == null
                    ? future.get()
                    : future.get(timeout.toMillis(), TimeUnit.MILLISECONDS);
        } catch (TimeoutException e) {
            future.cancel(false);
            String message = what + " did not answer within " + timeout;
            if (config.timeoutAction() == IsolationConfig.TimeoutAction.KILL && isAlive()) {
                kill(message);
                throw new IsolatedTimeoutException(message + "; the isolated JVM was killed");
            }
            throw new IsolatedTimeoutException(message);
        } catch (InterruptedException e) {
            future.cancel(false);
            Thread.currentThread().interrupt();
            throw new IsolatedException("Interrupted while waiting for " + what, e);
        } catch (ExecutionException e) {
            if (e.getCause() instanceof IsolatedException ie) {
                throw ie;
            }
            throw new IsolatedException(what + " failed", e.getCause());
        }
    }

    // ------------------------------------------------------------------ state

    public long pid() {
        return process.pid();
    }

    public boolean isAlive() {
        return death.get() == null && process.isAlive();
    }

    /** Where the child's logs, hs_err files and socket are. */
    public Path workDirectory() {
        return workDir;
    }

    public IsolationConfig config() {
        return config;
    }

    /** Completes once the child is gone, with the exception calls failed with. */
    public CompletableFuture<IsolatedException> terminated() {
        return terminated;
    }

    // ------------------------------------------------------------------ shutdown

    /** Asks the child to exit, then forces it; pending calls fail. Idempotent. */
    @Override
    public void close() {
        if (closing) {
            return;
        }
        closing = true;
        outbox.add(Frame.of(FrameType.GOODBYE, 0));
        try {
            if (!process.waitFor(5, TimeUnit.SECONDS)) {
                process.destroy();
                if (!process.waitFor(2, TimeUnit.SECONDS)) {
                    process.destroyForcibly().waitFor(5, TimeUnit.SECONDS);
                }
            }
        } catch (InterruptedException e) {
            process.destroyForcibly();
            Thread.currentThread().interrupt();
        }
        die(new IsolatedException("The isolated JVM (pid " + pid() + ") was closed"));
        terminated.join();
        if (ownsWorkDir && findHsErr().isEmpty()) {
            deleteRecursively(workDir);
        }
    }

    /** Kills the child at once (SIGKILL), e.g. when it hangs in native code. */
    public void kill() {
        process.destroyForcibly();
    }

    private void kill(String reason) {
        killReason = reason;
        process.destroyForcibly();
    }

    // ------------------------------------------------------------------ threads

    private void writeLoop() {
        try {
            while (true) {
                Frame frame = outbox.take();
                io.write(channel, frame);
                if (frame.type() == FrameType.GOODBYE) {
                    return;
                }
            }
        } catch (InterruptedException e) {
            // dying
        } catch (IOException e) {
            if (!closing) {
                connectionLost(e);
            }
        }
    }

    private void readLoop() {
        try {
            while (true) {
                Frame frame = io.read(channel);
                switch (frame.type()) {
                    case HELLO -> complete(0L, frame.payload());
                    case PONG -> lastPongNanos = System.nanoTime();
                    case RESPONSE -> complete(frame.callId(), frame.payload());
                    case ERROR -> {
                        Pending p = pending.remove(frame.callId());
                        if (p != null) {
                            Wire.Reader r = Wire.reader(frame.payload());
                            p.future.completeExceptionally(
                                    new IsolatedRemoteException(
                                            r.string(), r.string(), r.string()));
                        }
                    }
                    case STREAM -> {
                        Pending p = pending.get(frame.callId());
                        if (p != null && p.onItem != null) {
                            try {
                                p.onItem.accept(frame.payload());
                            } catch (RuntimeException e) {
                                // the consumer gave up: stop the producer in the child
                                if (pending.remove(frame.callId()) != null) {
                                    outbox.add(Frame.of(FrameType.CANCEL, frame.callId()));
                                }
                                p.future.completeExceptionally(e);
                            }
                        }
                    }
                    case GOODBYE -> {
                        return;
                    }
                    default -> throw new IOException("Unexpected frame from the child: " + frame);
                }
            }
        } catch (IOException e) {
            if (!closing) {
                connectionLost(e);
            }
        }
    }

    private void complete(long callId, byte[] payload) {
        Pending p = pending.remove(callId);
        if (p != null) {
            p.future.complete(payload);
        }
    }

    /** The socket broke; if the process is still there it is in an unknown state: kill it. */
    private void connectionLost(IOException cause) {
        if (process.isAlive()) {
            try {
                // give a crashing JVM a moment to finish dying, so the report has its exit code
                process.waitFor(2, TimeUnit.SECONDS);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
        if (process.isAlive()) {
            kill("lost its connection: " + cause);
            die(new IsolatedCrashException(report(-1, "lost its connection: " + cause)));
        }
        // otherwise onProcessExit reports the death with the exit code
    }

    private void onProcessExit() {
        if (closing) {
            die(new IsolatedException("The isolated JVM (pid " + pid() + ") was closed"));
        } else {
            String reason = killReason;
            die(
                    new IsolatedCrashException(
                            report(
                                    process.exitValue(),
                                    reason == null ? "died" : "was killed: " + reason)));
        }
    }

    private void die(IsolatedException cause) {
        if (!death.compareAndSet(null, cause)) {
            return;
        }
        for (Long id : List.copyOf(pending.keySet())) {
            Pending p = pending.remove(id);
            if (p != null) {
                p.future.completeExceptionally(cause);
            }
        }
        writer.interrupt();
        try {
            channel.close();
        } catch (IOException e) {
            // ignore
        }
        terminated.complete(cause);
    }

    private CrashReport report(int exitCode, String reason) {
        return new CrashReport(pid(), exitCode, stderrTail(), findHsErr(), reason);
    }

    private Optional<Path> findHsErr() {
        Path hsErr = workDir.resolve("hs_err_pid" + pid() + ".log");
        return Files.exists(hsErr) ? Optional.of(hsErr) : Optional.empty();
    }

    private String stderrTail() {
        Path err = workDir.resolve("child.err");
        if (!Files.exists(err)) {
            return null;
        }
        try (RandomAccessFile f = new RandomAccessFile(err.toFile(), "r")) {
            long start = Math.max(0, f.length() - STDERR_TAIL_BYTES);
            byte[] bytes = new byte[(int) (f.length() - start)];
            f.seek(start);
            f.readFully(bytes);
            return new String(bytes, StandardCharsets.UTF_8);
        } catch (IOException e) {
            return null;
        }
    }

    // ------------------------------------------------------------------ helpers

    private static Path createPrivateTempDir() throws IOException {
        if (FileSystems.getDefault().supportedFileAttributeViews().contains("posix")) {
            return Files.createTempDirectory(
                    "jota-iso",
                    PosixFilePermissions.asFileAttribute(
                            PosixFilePermissions.fromString("rwx------")));
        }
        return Files.createTempDirectory("jota-iso");
    }

    private static String newToken() {
        byte[] bytes = new byte[32];
        new SecureRandom().nextBytes(bytes);
        return HexFormat.of().formatHex(bytes);
    }

    private static void deleteRecursively(Path dir) {
        try (Stream<Path> paths = Files.walk(dir)) {
            paths.sorted(Comparator.reverseOrder())
                    .forEach(
                            p -> {
                                try {
                                    Files.deleteIfExists(p);
                                } catch (IOException e) {
                                    throw new UncheckedIOException(e);
                                }
                            });
        } catch (IOException | UncheckedIOException e) {
            // best effort
        }
    }

    @Override
    public String toString() {
        return "IsolatedJvm{pid=" + pid() + ", alive=" + isAlive() + ", dir=" + workDir + "}";
    }
}
