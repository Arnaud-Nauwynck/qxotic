package com.qxotic.jota.runtime.isolated.wire;

/** The kinds of frame exchanged between the parent JVM and the child JVM. */
public enum FrameType {
    /** child → parent, first frame: protocol version, child pid, authentication token. */
    HELLO(1),
    /** parent → child: service name, method name, payload. */
    REQUEST(2),
    /** child → parent: the final result of a call. */
    RESPONSE(3),
    /** child → parent: the call failed; remote class, message and stack. */
    ERROR(4),
    /** child → parent: one item of a streamed result, before the RESPONSE. */
    STREAM(5),
    /** parent → child: stop the call with this id. */
    CANCEL(6),
    /** either way: orderly shutdown. */
    GOODBYE(7),
    /**
     * parent → child: liveness probe, answered by the child's I/O thread (not a worker), so it
     * works while every worker is busy in a long native call.
     */
    PING(8),
    /** child → parent: the answer to a PING, with the same id. */
    PONG(9);

    private final byte code;

    FrameType(int code) {
        this.code = (byte) code;
    }

    public byte code() {
        return code;
    }

    public static FrameType of(byte code) {
        for (FrameType type : values()) {
            if (type.code == code) {
                return type;
            }
        }
        throw new IllegalArgumentException("Unknown frame type: " + code);
    }
}
