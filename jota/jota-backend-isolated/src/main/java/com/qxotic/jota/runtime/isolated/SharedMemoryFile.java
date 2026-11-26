package com.qxotic.jota.runtime.isolated;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Objects;

/**
 * A file mapped into memory, shared by the parent and the child for bulk data (tensors, logits)
 * that should not be copied through frames. Payloads name it by {@link #path()} plus an offset and
 * a length.
 *
 * <p>On Linux the default directory is {@code /dev/shm} (RAM, no disk I/O). A child crash does not
 * invalidate the parent's mapping: the file outlives the process.
 */
public final class SharedMemoryFile implements AutoCloseable {

    private final Path path;
    private final Arena arena;
    private final MemorySegment segment;
    private final boolean owner;

    private SharedMemoryFile(Path path, Arena arena, MemorySegment segment, boolean owner) {
        this.path = path;
        this.arena = arena;
        this.segment = segment;
        this.owner = owner;
    }

    /** Creates a new zero-filled file of {@code size} bytes in {@code dir} and maps it. */
    public static SharedMemoryFile create(Path dir, long size) {
        Objects.requireNonNull(dir, "dir");
        if (size <= 0) {
            throw new IllegalArgumentException("size must be > 0: " + size);
        }
        try {
            Path path = Files.createTempFile(dir, "jota-shm-", ".bin");
            return map(path, size, true);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /** Creates a file in the default shared directory. */
    public static SharedMemoryFile create(long size) {
        return create(defaultDirectory(), size);
    }

    /** Maps an existing file created by the other side; closing it does not delete the file. */
    public static SharedMemoryFile open(Path path) {
        try {
            return map(path, Files.size(path), false);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private static SharedMemoryFile map(Path path, long size, boolean create) throws IOException {
        Arena arena = Arena.ofShared();
        try (FileChannel ch =
                FileChannel.open(path, StandardOpenOption.READ, StandardOpenOption.WRITE)) {
            if (create) {
                ch.truncate(0);
                // sets the size: the mapping must never extend past the end of the file (SIGBUS)
                ch.write(java.nio.ByteBuffer.wrap(new byte[1]), size - 1);
            }
            MemorySegment segment = ch.map(FileChannel.MapMode.READ_WRITE, 0, size, arena);
            return new SharedMemoryFile(path, arena, segment, create);
        } catch (IOException | RuntimeException e) {
            arena.close();
            throw e;
        }
    }

    public static Path defaultDirectory() {
        Path shm = Path.of("/dev/shm");
        if (Files.isDirectory(shm) && Files.isWritable(shm)) {
            return shm;
        }
        return Path.of(System.getProperty("java.io.tmpdir"));
    }

    public Path path() {
        return path;
    }

    public long size() {
        return segment.byteSize();
    }

    /** The mapped bytes; valid until {@link #close()}. */
    public MemorySegment segment() {
        return segment;
    }

    /** Unmaps; the creator also deletes the file. */
    @Override
    public void close() {
        arena.close();
        if (owner) {
            try {
                Files.deleteIfExists(path);
            } catch (IOException e) {
                // best effort
            }
        }
    }
}
