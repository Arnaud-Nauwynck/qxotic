package com.qxotic.jota.runtime.xla;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.Optional;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Serialized executables on disk ({@code PJRT_Executable_Serialize}), one file per program key. The
 * format belongs to the plugin build, so the directory is named after a fingerprint of the plugin
 * (path, size, modification time, C API and platform versions): another plugin never reads these
 * files. I/O failures only disable the store for the rest of the run.
 */
final class XlaExecutableStore {

    private static final Logger LOG = Logger.getLogger(XlaExecutableStore.class.getName());

    private final Path dir;
    private volatile boolean enabled;

    /** A store under {@code root}, or a disabled one when {@code root} is null. */
    XlaExecutableStore(Path root, String pluginFingerprint) {
        this.dir = root == null ? null : root.resolve(pluginFingerprint);
        this.enabled = root != null;
    }

    static XlaExecutableStore disabled() {
        return new XlaExecutableStore(null, "");
    }

    boolean isEnabled() {
        return enabled;
    }

    Path directory() {
        return dir;
    }

    Optional<byte[]> load(String key) {
        if (!enabled) {
            return Optional.empty();
        }
        Path file = file(key);
        if (!Files.isRegularFile(file)) {
            return Optional.empty();
        }
        try {
            return Optional.of(Files.readAllBytes(file));
        } catch (IOException e) {
            disable("cannot read " + file, e);
            return Optional.empty();
        }
    }

    void store(String key, byte[] bytes) {
        if (!enabled) {
            return;
        }
        Path file = file(key);
        try {
            Files.createDirectories(dir);
            Path tmp = Files.createTempFile(dir, key, ".tmp");
            Files.write(tmp, bytes);
            try {
                Files.move(tmp, file, StandardCopyOption.ATOMIC_MOVE);
            } catch (AtomicMoveNotSupportedException e) {
                Files.move(tmp, file, StandardCopyOption.REPLACE_EXISTING);
            }
        } catch (IOException e) {
            disable("cannot write " + file, e);
        }
    }

    /** Removes an entry the plugin refused to load. */
    void delete(String key) {
        if (!enabled) {
            return;
        }
        try {
            Files.deleteIfExists(file(key));
        } catch (IOException e) {
            disable("cannot delete " + file(key), e);
        }
    }

    void disable(String reason, Throwable cause) {
        if (enabled) {
            enabled = false;
            LOG.log(Level.WARNING, "XLA executable disk cache disabled: " + reason, cause);
        }
    }

    private Path file(String key) {
        return dir.resolve(key + ".xlaexe");
    }

    /** A short stable fingerprint of the plugin build and its C API / platform versions. */
    static String fingerprint(Path library, String apiVersion, String platformVersion) {
        StringBuilder sb = new StringBuilder();
        Path abs = library.toAbsolutePath().normalize();
        sb.append(abs).append('\n').append(apiVersion).append('\n').append(platformVersion);
        try {
            sb.append('\n')
                    .append(Files.size(abs))
                    .append('\n')
                    .append(Files.getLastModifiedTime(abs).toMillis());
        } catch (IOException e) {
            // the plugin was loaded from this path: size / time are a bonus, not required
        }
        try {
            byte[] digest =
                    MessageDigest.getInstance("SHA-256")
                            .digest(sb.toString().getBytes(StandardCharsets.UTF_8));
            return HexFormat.of().formatHex(digest, 0, 8);
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException(e);
        }
    }
}
