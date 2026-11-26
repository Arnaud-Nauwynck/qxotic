package com.qxotic.jota.runtime.isolated.spi;

/** What a running call can do besides returning: stream items, notice cancellation. */
public interface CallContext {

    /** The id of the call, unique per connection. */
    long callId();

    /**
     * Sends one item of a streamed result to the parent (e.g. a generated token), before the final
     * response.
     */
    void emit(byte[] item);

    /** True once the parent cancelled this call; long loops should check it and stop. */
    boolean isCancelled();
}
