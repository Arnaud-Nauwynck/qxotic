package com.qxotic.jota.runtime.iree;

import com.qxotic.jota.runtime.KernelCachePaths;
import com.qxotic.jota.runtime.NativeLibraryLoader;
import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

/**
 * A native library packaged in this jar under {@code META-INF/native/<os>/<arch>/}, extracted to a
 * file because FFM ({@code SymbolLookup.libraryLookup}) loads from a path, not from a resource.
 *
 * <p>The file goes to {@code <KernelCachePaths.versionRoot()>/iree/lib/<sha256>/}: named by its
 * content, extracted once and shared by later JVMs.
 */
final class PackagedLibrary {

    private static final Map<String, Optional<Path>> EXTRACTED = new ConcurrentHashMap<>();

    private PackagedLibrary() {}

    /** The extracted library {@code name} (e.g. {@code jota_iree_rt}), or empty if not packaged. */
    static Optional<Path> find(String name) {
        return EXTRACTED.computeIfAbsent(name, PackagedLibrary::extract);
    }

    private static Optional<Path> extract(String name) {
        if (!NativeLibraryLoader.isPlatformSupported()) {
            return Optional.empty();
        }
        String platform = NativeLibraryLoader.currentPlatform(); // e.g. linux-x86_64
        int dash = platform.indexOf('-');
        String file = System.mapLibraryName(name);
        String resource =
                "/META-INF/native/"
                        + platform.substring(0, dash)
                        + "/"
                        + platform.substring(dash + 1)
                        + "/"
                        + file;
        byte[] bytes;
        try (InputStream in = PackagedLibrary.class.getResourceAsStream(resource)) {
            if (in == null) {
                return Optional.empty();
            }
            bytes = in.readAllBytes();
        } catch (IOException e) {
            throw new UncheckedIOException("Cannot read " + resource, e);
        }
        Path dir =
                KernelCachePaths.versionRoot()
                        .resolve("iree")
                        .resolve("lib")
                        .resolve(sha256(bytes));
        Path target = dir.resolve(file);
        try {
            if (!Files.isRegularFile(target) || Files.size(target) != bytes.length) {
                Files.createDirectories(dir);
                Path tmp = Files.createTempFile(dir, file, ".tmp");
                Files.write(tmp, bytes);
                Files.move(
                        tmp,
                        target,
                        StandardCopyOption.REPLACE_EXISTING,
                        StandardCopyOption.ATOMIC_MOVE);
            }
        } catch (IOException e) {
            throw new UncheckedIOException("Cannot extract " + resource + " to " + target, e);
        }
        return Optional.of(target);
    }

    private static String sha256(byte[] bytes) {
        try {
            return HexFormat.of()
                    .formatHex(MessageDigest.getInstance("SHA-256").digest(bytes), 0, 16);
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException(e);
        }
    }
}
