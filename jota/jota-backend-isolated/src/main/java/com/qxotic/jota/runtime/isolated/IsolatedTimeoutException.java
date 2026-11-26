package com.qxotic.jota.runtime.isolated;

/** A call took longer than its timeout; it was cancelled in the child (best effort). */
public final class IsolatedTimeoutException extends IsolatedException {

    public IsolatedTimeoutException(String message) {
        super(message);
    }
}
