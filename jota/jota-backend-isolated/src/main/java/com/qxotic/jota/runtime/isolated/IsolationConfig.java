package com.qxotic.jota.runtime.isolated;

import com.qxotic.jota.runtime.isolated.server.IsolatedServerMain;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * How to launch and talk to the child JVM. Immutable; build with {@link #builder()}.
 *
 * <p>Defaults: the parent's own {@code java} and classpath, {@code
 * --enable-native-access=ALL-UNNAMED}, one worker thread in the child, 30 s to start, no call
 * timeout (a timed-out call is only cancelled), no heartbeat, frames up to 256 MiB, child output to
 * files in its work directory.
 */
public record IsolationConfig(
        Path javaExecutable,
        String classpath,
        List<String> jvmOptions,
        Map<String, String> systemProperties,
        Map<String, String> environment,
        String mainClass,
        Duration startupTimeout,
        Duration callTimeout,
        TimeoutAction timeoutAction,
        Duration heartbeatInterval,
        Duration heartbeatTimeout,
        int maxFrameBytes,
        int serverThreads,
        boolean inheritOutput,
        Path workDirectory) {

    public IsolationConfig {
        Objects.requireNonNull(javaExecutable, "javaExecutable");
        Objects.requireNonNull(classpath, "classpath");
        jvmOptions = List.copyOf(jvmOptions);
        systemProperties = Map.copyOf(systemProperties);
        environment = Map.copyOf(environment);
        Objects.requireNonNull(mainClass, "mainClass");
        Objects.requireNonNull(startupTimeout, "startupTimeout");
        Objects.requireNonNull(timeoutAction, "timeoutAction");
        if (heartbeatInterval != null) {
            if (heartbeatInterval.isNegative() || heartbeatInterval.isZero()) {
                throw new IllegalArgumentException("heartbeatInterval <= 0: " + heartbeatInterval);
            }
            if (heartbeatTimeout == null || heartbeatTimeout.compareTo(heartbeatInterval) <= 0) {
                throw new IllegalArgumentException(
                        "heartbeatTimeout must be longer than heartbeatInterval: "
                                + heartbeatTimeout
                                + " <= "
                                + heartbeatInterval);
            }
        }
        if (maxFrameBytes < 1024) {
            throw new IllegalArgumentException("maxFrameBytes < 1024: " + maxFrameBytes);
        }
        if (serverThreads < 1) {
            throw new IllegalArgumentException("serverThreads < 1: " + serverThreads);
        }
    }

    /** What a call timeout does besides failing the call. */
    public enum TimeoutAction {
        /** Send a cancel to the child and fail the call; the child stays (default). */
        CANCEL,
        /**
         * Kill the child: a native call that does not return (a hung driver) never sees the cancel
         * flag, and would keep its worker thread forever. Other pending calls fail with {@link
         * IsolatedCrashException}; a supervisor restarts the child on the next call.
         */
        KILL
    }

    public static IsolationConfig defaults() {
        return builder().build();
    }

    public static Builder builder() {
        return new Builder();
    }

    public Builder toBuilder() {
        Builder b = new Builder();
        b.javaExecutable = javaExecutable;
        b.classpath = classpath;
        b.jvmOptions = new ArrayList<>(jvmOptions);
        b.systemProperties = new LinkedHashMap<>(systemProperties);
        b.environment = new LinkedHashMap<>(environment);
        b.mainClass = mainClass;
        b.startupTimeout = startupTimeout;
        b.callTimeout = callTimeout;
        b.timeoutAction = timeoutAction;
        b.heartbeatInterval = heartbeatInterval;
        b.heartbeatTimeout = heartbeatTimeout;
        b.maxFrameBytes = maxFrameBytes;
        b.serverThreads = serverThreads;
        b.inheritOutput = inheritOutput;
        b.workDirectory = workDirectory;
        return b;
    }

    /** The full command line of the child, without the trailing socket path argument. */
    List<String> command() {
        List<String> cmd = new ArrayList<>();
        cmd.add(javaExecutable.toString());
        cmd.addAll(jvmOptions);
        systemProperties.forEach((k, v) -> cmd.add("-D" + k + "=" + v));
        cmd.add("-D" + IsolatedServerMain.THREADS_PROPERTY + "=" + serverThreads);
        cmd.add("-D" + IsolatedServerMain.MAX_FRAME_PROPERTY + "=" + maxFrameBytes);
        cmd.add("-cp");
        cmd.add(classpath);
        cmd.add(mainClass);
        return cmd;
    }

    public static final class Builder {
        private Path javaExecutable =
                Path.of(System.getProperty("java.home"), "bin", isWindows() ? "java.exe" : "java");
        private String classpath = System.getProperty("java.class.path");
        private List<String> jvmOptions =
                new ArrayList<>(List.of("--enable-native-access=ALL-UNNAMED"));
        private Map<String, String> systemProperties = new LinkedHashMap<>();
        private Map<String, String> environment = new LinkedHashMap<>();
        private String mainClass = IsolatedServerMain.class.getName();
        private Duration startupTimeout = Duration.ofSeconds(30);
        private Duration callTimeout = null;
        private TimeoutAction timeoutAction = TimeoutAction.CANCEL;
        private Duration heartbeatInterval = null;
        private Duration heartbeatTimeout = null;
        private int maxFrameBytes = 256 << 20;
        private int serverThreads = 1;
        private boolean inheritOutput = false;
        private Path workDirectory = null;

        private Builder() {}

        public Builder javaExecutable(Path java) {
            this.javaExecutable = java;
            return this;
        }

        public Builder classpath(String classpath) {
            this.classpath = classpath;
            return this;
        }

        /** Replaces the JVM options (heap size, GC, {@code --enable-native-access}...). */
        public Builder jvmOptions(List<String> options) {
            this.jvmOptions = new ArrayList<>(options);
            return this;
        }

        public Builder addJvmOption(String option) {
            this.jvmOptions.add(option);
            return this;
        }

        public Builder systemProperty(String key, String value) {
            this.systemProperties.put(key, value);
            return this;
        }

        /**
         * Extra environment of the child, e.g. {@code LD_LIBRARY_PATH}, {@code
         * CUDA_VISIBLE_DEVICES}.
         */
        public Builder environment(String key, String value) {
            this.environment.put(key, value);
            return this;
        }

        public Builder mainClass(String mainClass) {
            this.mainClass = mainClass;
            return this;
        }

        public Builder startupTimeout(Duration timeout) {
            this.startupTimeout = timeout;
            return this;
        }

        /** Default timeout of a call; {@code null} waits forever. */
        public Builder callTimeout(Duration timeout) {
            this.callTimeout = timeout;
            return this;
        }

        /** What a call timeout does besides failing the call; default {@code CANCEL}. */
        public Builder timeoutAction(TimeoutAction action) {
            this.timeoutAction = action;
            return this;
        }

        /**
         * Pings the child every {@code interval} and kills it when no answer came for {@code
         * timeout}: catches a child that is alive but wedged (stuck in a GC spiral, stopped,
         * deadlocked JVM). The ping is answered by the child's I/O thread, so long calls do not
         * trip it. {@code null} interval: no heartbeat (default).
         */
        public Builder heartbeat(Duration interval, Duration timeout) {
            this.heartbeatInterval = interval;
            this.heartbeatTimeout = timeout;
            return this;
        }

        public Builder maxFrameBytes(int bytes) {
            this.maxFrameBytes = bytes;
            return this;
        }

        /** Worker threads in the child; 1 keeps native libraries single-threaded. */
        public Builder serverThreads(int threads) {
            this.serverThreads = threads;
            return this;
        }

        /** True: the child writes on the parent's stdout / stderr; false: on files. */
        public Builder inheritOutput(boolean inherit) {
            this.inheritOutput = inherit;
            return this;
        }

        /** Where the socket, logs and hs_err files go; {@code null}: a fresh temp directory. */
        public Builder workDirectory(Path dir) {
            this.workDirectory = dir;
            return this;
        }

        public IsolationConfig build() {
            return new IsolationConfig(
                    javaExecutable,
                    classpath,
                    jvmOptions,
                    systemProperties,
                    environment,
                    mainClass,
                    startupTimeout,
                    callTimeout,
                    timeoutAction,
                    heartbeatInterval,
                    heartbeatTimeout,
                    maxFrameBytes,
                    serverThreads,
                    inheritOutput,
                    workDirectory);
        }
    }

    static boolean isWindows() {
        return System.getProperty("os.name", "").toLowerCase(java.util.Locale.ROOT).contains("win");
    }
}
