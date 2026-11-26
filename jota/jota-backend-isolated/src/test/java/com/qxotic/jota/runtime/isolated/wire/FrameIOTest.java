package com.qxotic.jota.runtime.isolated.wire;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.nio.channels.Channels;
import java.util.Arrays;
import org.junit.jupiter.api.Test;

class FrameIOTest {

    @Test
    void roundTripsFrames() throws IOException {
        FrameIO io = new FrameIO(1 << 20);
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        var out = Channels.newChannel(bytes);
        io.write(out, new Frame(FrameType.REQUEST, 42, new byte[] {1, 2, 3}));
        io.write(out, Frame.of(FrameType.GOODBYE, 0));

        var in = Channels.newChannel(new ByteArrayInputStream(bytes.toByteArray()));
        Frame first = io.read(in);
        assertEquals(FrameType.REQUEST, first.type());
        assertEquals(42, first.callId());
        assertArrayEquals(new byte[] {1, 2, 3}, first.payload());
        Frame second = io.read(in);
        assertEquals(FrameType.GOODBYE, second.type());
        assertEquals(0, second.payload().length);
        assertThrows(EOFException.class, () -> io.read(in));
    }

    @Test
    void rejectsOversizedFrames() throws IOException {
        FrameIO small = new FrameIO(64);
        var out = Channels.newChannel(new ByteArrayOutputStream());
        assertThrows(
                IOException.class,
                () -> small.write(out, new Frame(FrameType.REQUEST, 1, new byte[100])));

        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        new FrameIO(1 << 20)
                .write(Channels.newChannel(bytes), new Frame(FrameType.REQUEST, 1, new byte[100]));
        var in = Channels.newChannel(new ByteArrayInputStream(bytes.toByteArray()));
        assertThrows(IOException.class, () -> small.read(in));
    }

    @Test
    void truncatedFrameIsEof() throws IOException {
        FrameIO io = new FrameIO(1 << 20);
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        io.write(Channels.newChannel(bytes), new Frame(FrameType.STREAM, 7, new byte[10]));
        byte[] cut = Arrays.copyOf(bytes.toByteArray(), bytes.size() - 3);
        var in = Channels.newChannel(new ByteArrayInputStream(cut));
        assertThrows(EOFException.class, () -> io.read(in));
    }

    @Test
    void wireRoundTrip() {
        byte[] payload =
                Wire.writer()
                        .string("héllo")
                        .string(null)
                        .i32(-5)
                        .i64(Long.MAX_VALUE)
                        .f64(1.5)
                        .bool(true)
                        .bytes(new byte[] {9})
                        .toByteArray();
        Wire.Reader r = Wire.reader(payload);
        assertEquals("héllo", r.string());
        assertNull(r.string());
        assertEquals(-5, r.i32());
        assertEquals(Long.MAX_VALUE, r.i64());
        assertEquals(1.5, r.f64());
        assertEquals(true, r.bool());
        assertArrayEquals(new byte[] {9}, r.bytes());
        assertFalse(r.hasMore());
        assertThrows(IllegalArgumentException.class, r::i32);
    }
}
