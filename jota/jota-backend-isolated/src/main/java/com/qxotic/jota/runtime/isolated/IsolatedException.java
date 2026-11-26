package com.qxotic.jota.runtime.isolated;

/** Base of the failures a call to the child JVM can end with. */
public class IsolatedException extends RuntimeException {

    public IsolatedException(String message) {
        super(message);
    }

    public IsolatedException(String message, Throwable cause) {
        super(message, cause);
    }
}
