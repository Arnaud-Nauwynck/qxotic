package com.qxotic.jota.runtime.isolated.wire;

import java.util.Objects;

/**
 * One protocol frame: a type, the id of the call it belongs to (0 for connection-level frames) and
 * an opaque payload.
 */
public record Frame(FrameType type, long callId, byte[] payload) {

    private static final byte[] EMPTY = new byte[0];

    public Frame {
        Objects.requireNonNull(type, "type");
        payload = payload == null ? EMPTY : payload;
    }

    public static Frame of(FrameType type, long callId) {
        return new Frame(type, callId, EMPTY);
    }

    @Override
    public String toString() {
        return "Frame{" + type + ", callId=" + callId + ", " + payload.length + " bytes}";
    }
}
