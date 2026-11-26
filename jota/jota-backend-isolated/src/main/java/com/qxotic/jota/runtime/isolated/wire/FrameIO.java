package com.qxotic.jota.runtime.isolated.wire;

import java.io.EOFException;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.ReadableByteChannel;
import java.nio.channels.WritableByteChannel;

/**
 * Reads and writes frames on a byte channel.
 *
 * <p>Layout, big-endian: {@code int length | byte type | long callId | payload}, where {@code
 * length} counts the bytes after itself (9 + payload length). Writers must be serialized by the
 * caller; a single reader thread is assumed.
 */
public final class FrameIO {

    /** Protocol version sent in HELLO; bump on any incompatible change. */
    public static final int PROTOCOL_VERSION = 1;

    static final int HEADER = Byte.BYTES + Long.BYTES;

    private final int maxFrameBytes;

    public FrameIO(int maxFrameBytes) {
        if (maxFrameBytes < HEADER) {
            throw new IllegalArgumentException("maxFrameBytes too small: " + maxFrameBytes);
        }
        this.maxFrameBytes = maxFrameBytes;
    }

    public int maxFrameBytes() {
        return maxFrameBytes;
    }

    public void write(WritableByteChannel channel, Frame frame) throws IOException {
        long length = (long) HEADER + frame.payload().length;
        if (length > maxFrameBytes) {
            throw new IOException(
                    "Frame too large: "
                            + frame.payload().length
                            + " payload bytes, limit "
                            + (maxFrameBytes - HEADER));
        }
        ByteBuffer header = ByteBuffer.allocate(Integer.BYTES + HEADER).order(ByteOrder.BIG_ENDIAN);
        header.putInt((int) length).put(frame.type().code()).putLong(frame.callId()).flip();
        writeFully(channel, header);
        writeFully(channel, ByteBuffer.wrap(frame.payload()));
    }

    /**
     * Reads the next frame.
     *
     * @throws EOFException when the peer closed the channel (cleanly between two frames or not)
     */
    public Frame read(ReadableByteChannel channel) throws IOException {
        ByteBuffer lengthBuffer = ByteBuffer.allocate(Integer.BYTES);
        readFully(channel, lengthBuffer);
        int length = lengthBuffer.flip().getInt();
        if (length < HEADER || length > maxFrameBytes) {
            throw new IOException("Corrupt or oversized frame: length " + length);
        }
        ByteBuffer body = ByteBuffer.allocate(length);
        readFully(channel, body);
        body.flip();
        FrameType type = FrameType.of(body.get());
        long callId = body.getLong();
        byte[] payload = new byte[body.remaining()];
        body.get(payload);
        return new Frame(type, callId, payload);
    }

    private static void writeFully(WritableByteChannel channel, ByteBuffer buffer)
            throws IOException {
        while (buffer.hasRemaining()) {
            channel.write(buffer);
        }
    }

    private static void readFully(ReadableByteChannel channel, ByteBuffer buffer)
            throws IOException {
        while (buffer.hasRemaining()) {
            if (channel.read(buffer) < 0) {
                throw new EOFException("Channel closed");
            }
        }
    }
}
