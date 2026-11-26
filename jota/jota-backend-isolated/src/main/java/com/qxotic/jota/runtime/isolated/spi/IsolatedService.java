package com.qxotic.jota.runtime.isolated.spi;

/**
 * A service hosted by the child JVM, found there with {@link java.util.ServiceLoader}.
 *
 * <p>This is where the risky work goes: loading a model, calling a native backend, generating
 * tokens. The parent reaches it by {@link #name()} and a method name; payloads are opaque bytes
 * (see {@link com.qxotic.jota.runtime.isolated.wire.Wire}).
 *
 * <p>Implementations must have a public no-argument constructor and be listed in {@code
 * META-INF/services/com.qxotic.jota.runtime.isolated.spi.IsolatedService} on the child classpath.
 */
public interface IsolatedService {

    /** The name callers use, unique in the child; e.g. {@code "llm"}, {@code "jota.device"}. */
    String name();

    /**
     * Runs one call.
     *
     * @param method the method name sent by the parent
     * @param payload the request bytes
     * @param context streaming and cancellation for this call
     * @return the response bytes (may be empty, not null)
     * @throws Exception sent back to the parent as an {@code IsolatedRemoteException}
     */
    byte[] invoke(String method, byte[] payload, CallContext context) throws Exception;

    /** Called once when the child shuts down cleanly. */
    default void close() {}
}
