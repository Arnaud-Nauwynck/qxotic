package com.qxotic.jota.runtime.iree.runtime;

/** A failure reported by the IREE runtime, with its {@code iree_status_code_t} when known. */
public class IreeRuntimeException extends RuntimeException {

    /** {@code IREE_STATUS_*} code, or -1 when the failure did not come from an IREE status. */
    private final int statusCode;

    public IreeRuntimeException(String message) {
        super(message);
        this.statusCode = -1;
    }

    public IreeRuntimeException(String message, Throwable cause) {
        super(message, cause);
        this.statusCode = -1;
    }

    public IreeRuntimeException(String message, int statusCode) {
        super(message);
        this.statusCode = statusCode;
    }

    public int statusCode() {
        return statusCode;
    }
}
