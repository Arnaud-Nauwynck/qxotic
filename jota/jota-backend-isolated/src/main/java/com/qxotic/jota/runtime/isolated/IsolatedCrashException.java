package com.qxotic.jota.runtime.isolated;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;

/**
 * The child JVM died (native crash, {@code abort()}, halt, killed) or the connection to it broke
 * while calls were pending. The parent JVM is unaffected; the child must be restarted.
 */
public final class IsolatedCrashException extends IsolatedException {

    private final CrashReport report;

    public IsolatedCrashException(CrashReport report) {
        super(report.summary());
        this.report = report;
    }

    public CrashReport report() {
        return report;
    }

    /**
     * What is known about the death of a child.
     *
     * @param pid the child's process id
     * @param exitCode the exit code, or -1 if the process is still running (broken connection)
     * @param stderrTail the last lines the child wrote on stderr, if captured
     * @param hsErrFile the HotSpot fatal error log, when the JVM wrote one
     */
    public record CrashReport(
            long pid, int exitCode, String stderrTail, Optional<Path> hsErrFile, String reason) {

        /**
         * The header of the HotSpot fatal error log: the fault (e.g. {@code SIGSEGV (0xb) at
         * pc=...}) and the problematic frame (e.g. {@code C [libfoo.so+0x1234] bar+0x14}). This is
         * where the real cause is: with core dumps off, the JVM exits with code 1 after a fault.
         */
        public Optional<String> fatalError() {
            if (hsErrFile.isEmpty()) {
                return Optional.empty();
            }
            try (var lines = Files.lines(hsErrFile.get(), StandardCharsets.UTF_8)) {
                List<String> header = lines.limit(40).toList();
                StringBuilder sb = new StringBuilder();
                for (int i = 0; i < header.size(); i++) {
                    String line = header.get(i);
                    if (line.matches(
                                    "#\\s+(SIG[A-Z]+|EXCEPTION_[A-Z_]+|Internal Error|Out of"
                                            + " Memory).*")
                            || line.startsWith("# There is insufficient memory")) {
                        sb.append(line.substring(1).strip()).append('\n');
                    } else if (line.startsWith("# Problematic frame:") && i + 1 < header.size()) {
                        sb.append("problematic frame: ")
                                .append(header.get(i + 1).substring(1).strip())
                                .append('\n');
                    }
                }
                return sb.isEmpty() ? Optional.empty() : Optional.of(sb.toString().strip());
            } catch (IOException | UncheckedIOException e) {
                return Optional.empty();
            }
        }

        /** The signal that killed the child, decoded from {@code 128 + signo} on Unix. */
        public Optional<String> signal() {
            if (exitCode <= 128 || exitCode > 128 + 64 || IsolationConfig.isWindows()) {
                return Optional.empty();
            }
            int signo = exitCode - 128;
            String name =
                    switch (signo) {
                        case 4 -> "SIGILL";
                        case 6 -> "SIGABRT";
                        case 7 -> "SIGBUS";
                        case 8 -> "SIGFPE";
                        case 9 -> "SIGKILL";
                        case 11 -> "SIGSEGV";
                        case 15 -> "SIGTERM";
                        default -> "signal " + signo;
                    };
            return Optional.of(name);
        }

        public String summary() {
            StringBuilder sb = new StringBuilder("Isolated JVM (pid ").append(pid).append(") ");
            sb.append(reason);
            if (exitCode >= 0) {
                sb.append(", exit code ").append(exitCode);
                signal().ifPresent(s -> sb.append(" (").append(s).append(')'));
            }
            hsErrFile.ifPresent(f -> sb.append(", fatal error log ").append(f));
            fatalError().ifPresent(f -> sb.append('\n').append(f));
            if (stderrTail != null && !stderrTail.isBlank()) {
                sb.append("\n--- child stderr (tail) ---\n").append(stderrTail);
            }
            return sb.toString();
        }
    }
}
