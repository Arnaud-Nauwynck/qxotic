package com.qxotic.jota.runtime.isolated.server;

import com.qxotic.jota.runtime.isolated.spi.CallContext;
import com.qxotic.jota.runtime.isolated.spi.IsolatedService;
import com.qxotic.jota.runtime.isolated.wire.Frame;
import com.qxotic.jota.runtime.isolated.wire.FrameIO;
import com.qxotic.jota.runtime.isolated.wire.FrameType;
import com.qxotic.jota.runtime.isolated.wire.Wire;
import java.io.EOFException;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.net.StandardProtocolFamily;
import java.net.UnixDomainSocketAddress;
import java.nio.channels.SocketChannel;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.ServiceLoader;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Entry point of the child JVM: connects to the parent's socket, says HELLO, then runs the calls it
 * receives on the services found with {@link ServiceLoader}.
 *
 * <p>Usage: {@code java ... IsolatedServerMain <socket path>}, with the authentication token in the
 * {@value #TOKEN_ENV} environment variable. The child halts as soon as the parent goes away (socket
 * closed or parent process gone), so it never outlives it.
 */
public final class IsolatedServerMain {

    public static final String TOKEN_ENV = "JOTA_ISOLATED_TOKEN";
    public static final String THREADS_PROPERTY = "jota.isolated.serverThreads";
    public static final String MAX_FRAME_PROPERTY = "jota.isolated.maxFrameBytes";

    private final SocketChannel channel;
    private final FrameIO io;
    private final Map<String, IsolatedService> services;
    private final ExecutorService executor;
    private final Map<Long, Call> calls = new ConcurrentHashMap<>();
    private final LinkedBlockingQueue<Frame> outbox = new LinkedBlockingQueue<>();
    private volatile IOException writeFailure;

    private IsolatedServerMain(
            SocketChannel channel, FrameIO io, Map<String, IsolatedService> services, int threads) {
        this.channel = channel;
        this.io = io;
        this.services = services;
        // Platform threads: native libraries often keep per-thread state (CUDA contexts, TLS).
        this.executor =
                Executors.newFixedThreadPool(
                        threads,
                        r -> {
                            Thread t = new Thread(r, "isolated-worker");
                            t.setDaemon(true);
                            return t;
                        });
    }

    /**
     * One thread owns all writes: a worker interrupted while writing on the socket would close it
     * (it is an interruptible channel), and services may well be interrupted.
     */
    private void startWriter() {
        Thread writer =
                new Thread(
                        () -> {
                            try {
                                while (true) {
                                    io.write(channel, outbox.take());
                                }
                            } catch (IOException e) {
                                writeFailure = e;
                            } catch (InterruptedException e) {
                                // shutting down
                            }
                        },
                        "isolated-writer");
        writer.setDaemon(true);
        writer.start();
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            System.err.println("usage: IsolatedServerMain <socket path>");
            System.exit(2);
        }
        String token = System.getenv(TOKEN_ENV);
        if (token == null) {
            System.err.println(TOKEN_ENV + " not set");
            System.exit(2);
        }
        watchParent();
        int threads = Integer.getInteger(THREADS_PROPERTY, 1);
        FrameIO io = new FrameIO(Integer.getInteger(MAX_FRAME_PROPERTY, 256 << 20));
        SocketChannel channel = SocketChannel.open(StandardProtocolFamily.UNIX);
        channel.connect(UnixDomainSocketAddress.of(Path.of(args[0])));
        IsolatedServerMain server = new IsolatedServerMain(channel, io, loadServices(), threads);
        server.startWriter();
        server.hello(token);
        server.serve();
    }

    /** Parent JVM gone (even killed with SIGKILL) ⇒ this child goes too: no orphans. */
    private static void watchParent() {
        ProcessHandle.current()
                .parent()
                .ifPresent(parent -> parent.onExit().thenRun(() -> Runtime.getRuntime().halt(3)));
    }

    static Map<String, IsolatedService> loadServices() {
        Map<String, IsolatedService> services = new LinkedHashMap<>();
        services.put(SystemService.NAME, new SystemService());
        for (IsolatedService service : ServiceLoader.load(IsolatedService.class)) {
            IsolatedService previous = services.putIfAbsent(service.name(), service);
            if (previous != null && !(previous instanceof SystemService)) {
                System.err.println(
                        "isolated: duplicate service name '"
                                + service.name()
                                + "', keeping "
                                + previous.getClass().getName());
            }
        }
        return services;
    }

    private void hello(String token) throws IOException {
        byte[] payload =
                Wire.writer()
                        .i32(FrameIO.PROTOCOL_VERSION)
                        .i64(ProcessHandle.current().pid())
                        .string(token)
                        .toByteArray();
        send(new Frame(FrameType.HELLO, 0, payload));
    }

    private void serve() {
        try {
            while (true) {
                Frame frame = io.read(channel);
                switch (frame.type()) {
                    case REQUEST -> dispatch(frame);
                    case CANCEL -> {
                        Call call = calls.get(frame.callId());
                        if (call != null) {
                            // a flag, not an interrupt: interrupting native code is unsafe
                            call.cancelled.set(true);
                        }
                    }
                    case PING -> send(Frame.of(FrameType.PONG, frame.callId()));
                    case GOODBYE -> {
                        shutdown(0);
                        return;
                    }
                    default ->
                            System.err.println("isolated: unexpected frame from parent " + frame);
                }
            }
        } catch (EOFException e) {
            shutdown(0); // parent closed the connection
        } catch (Throwable t) {
            t.printStackTrace();
            shutdown(1);
        }
    }

    private void dispatch(Frame frame) {
        long callId = frame.callId();
        Wire.Reader reader = Wire.reader(frame.payload());
        String serviceName = reader.string();
        String method = reader.string();
        byte[] payload = reader.bytes();
        Call call = new Call(callId);
        calls.put(callId, call);
        executor.execute(
                () -> {
                    try {
                        IsolatedService service = services.get(serviceName);
                        if (service == null) {
                            throw new IllegalArgumentException(
                                    "No service '"
                                            + serviceName
                                            + "' in the child; known: "
                                            + services.keySet());
                        }
                        byte[] result = service.invoke(method, payload, call);
                        send(new Frame(FrameType.RESPONSE, callId, result));
                    } catch (Throwable t) {
                        sendError(callId, t);
                    } finally {
                        calls.remove(callId);
                    }
                });
    }

    private void sendError(long callId, Throwable t) {
        StringWriter stack = new StringWriter();
        t.printStackTrace(new PrintWriter(stack));
        byte[] payload =
                Wire.writer()
                        .string(t.getClass().getName())
                        .string(t.getMessage())
                        .string(stack.toString())
                        .toByteArray();
        try {
            send(new Frame(FrameType.ERROR, callId, payload));
        } catch (IOException e) {
            // the parent is gone; the read loop will notice
        }
    }

    private void send(Frame frame) throws IOException {
        IOException failure = writeFailure;
        if (failure != null) {
            throw new IOException("Connection to the parent is broken", failure);
        }
        if (frame.payload().length + 9L > io.maxFrameBytes()) {
            throw new IOException(
                    "Frame too large: "
                            + frame.payload().length
                            + " bytes, limit "
                            + io.maxFrameBytes());
        }
        outbox.add(frame);
    }

    private void shutdown(int status) {
        executor.shutdownNow();
        Thread closer =
                new Thread(
                        () -> services.values().forEach(IsolatedServerMain::closeQuietly),
                        "isolated-close");
        closer.setDaemon(true);
        closer.start();
        try {
            closer.join(TimeUnit.SECONDS.toMillis(5));
            executor.awaitTermination(1, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        // halt, not exit: a wedged native library must not keep the child alive in a hook.
        Runtime.getRuntime().halt(status);
    }

    private static void closeQuietly(IsolatedService service) {
        try {
            service.close();
        } catch (Throwable t) {
            System.err.println("isolated: closing service " + service.name() + " failed: " + t);
        }
    }

    private final class Call implements CallContext {
        final long id;
        final AtomicBoolean cancelled = new AtomicBoolean();

        Call(long id) {
            this.id = id;
        }

        @Override
        public long callId() {
            return id;
        }

        @Override
        public void emit(byte[] item) {
            try {
                send(new Frame(FrameType.STREAM, id, item));
            } catch (IOException e) {
                throw new java.io.UncheckedIOException(e);
            }
        }

        @Override
        public boolean isCancelled() {
            return cancelled.get();
        }
    }
}
