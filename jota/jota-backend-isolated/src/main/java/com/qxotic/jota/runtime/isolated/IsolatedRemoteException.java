package com.qxotic.jota.runtime.isolated;

/**
 * The service in the child threw: the child is still alive. The remote exception's class is only
 * named, never loaded or deserialized in the parent.
 */
public final class IsolatedRemoteException extends IsolatedException {

    private final String remoteClass;
    private final String remoteMessage;
    private final String remoteStackTrace;

    public IsolatedRemoteException(String remoteClass, String remoteMessage, String remoteStack) {
        super(remoteClass + (remoteMessage == null ? "" : ": " + remoteMessage));
        this.remoteClass = remoteClass;
        this.remoteMessage = remoteMessage;
        this.remoteStackTrace = remoteStack;
    }

    public String remoteClass() {
        return remoteClass;
    }

    public String remoteMessage() {
        return remoteMessage;
    }

    /** The child's {@code printStackTrace} output, causes included. */
    public String remoteStackTrace() {
        return remoteStackTrace;
    }
}
