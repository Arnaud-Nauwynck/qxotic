package com.qxotic.jota.runtime.isolated.wire;

import java.io.ByteArrayOutputStream;
import java.nio.BufferUnderflowException;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;

/**
 * A small typed encoding for payloads: strings, numbers, byte arrays, in a fixed order the two
 * sides agree on. Deliberately not Java serialization (fragile across versions, unsafe to read).
 */
public final class Wire {

    private Wire() {}

    public static Writer writer() {
        return new Writer();
    }

    public static Reader reader(byte[] bytes) {
        return new Reader(bytes);
    }

    public static byte[] utf8(String s) {
        return s.getBytes(StandardCharsets.UTF_8);
    }

    public static String utf8(byte[] b) {
        return new String(b, StandardCharsets.UTF_8);
    }

    /** Appends typed fields; {@link #toByteArray()} ends it. */
    public static final class Writer {
        private final ByteArrayOutputStream out = new ByteArrayOutputStream();
        private final ByteBuffer scratch = ByteBuffer.allocate(Long.BYTES);

        private Writer() {}

        public Writer string(String value) {
            return bytes(value == null ? null : utf8(value));
        }

        public Writer bytes(byte[] value) {
            if (value == null) {
                return i32(-1);
            }
            i32(value.length);
            out.writeBytes(value);
            return this;
        }

        public Writer i32(int value) {
            out.write(scratch.clear().putInt(value).array(), 0, Integer.BYTES);
            return this;
        }

        public Writer i64(long value) {
            out.write(scratch.clear().putLong(value).array(), 0, Long.BYTES);
            return this;
        }

        public Writer f64(double value) {
            return i64(Double.doubleToRawLongBits(value));
        }

        public Writer bool(boolean value) {
            return i32(value ? 1 : 0);
        }

        public byte[] toByteArray() {
            return out.toByteArray();
        }
    }

    /** Reads the fields back in the order they were written. */
    public static final class Reader {
        private final ByteBuffer in;

        private Reader(byte[] bytes) {
            this.in = ByteBuffer.wrap(bytes);
        }

        public String string() {
            byte[] b = bytes();
            return b == null ? null : utf8(b);
        }

        public byte[] bytes() {
            int n = i32();
            if (n < 0) {
                return null;
            }
            if (n > in.remaining()) {
                throw new IllegalArgumentException(
                        "Truncated payload: field of " + n + " bytes, " + in.remaining() + " left");
            }
            byte[] b = new byte[n];
            in.get(b);
            return b;
        }

        public int i32() {
            try {
                return in.getInt();
            } catch (BufferUnderflowException e) {
                throw new IllegalArgumentException("Truncated payload", e);
            }
        }

        public long i64() {
            try {
                return in.getLong();
            } catch (BufferUnderflowException e) {
                throw new IllegalArgumentException("Truncated payload", e);
            }
        }

        public double f64() {
            return Double.longBitsToDouble(i64());
        }

        public boolean bool() {
            return i32() != 0;
        }

        public boolean hasMore() {
            return in.hasRemaining();
        }
    }
}
