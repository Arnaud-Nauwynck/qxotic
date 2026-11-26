package com.qxotic.jota.runtime.xla.pjrt;

/** A {@code PJRT_Error} returned by a PJRT C API function. */
public final class PjrtException extends RuntimeException {

    private static final String[] CODES = {
        "OK",
        "CANCELLED",
        "UNKNOWN",
        "INVALID_ARGUMENT",
        "DEADLINE_EXCEEDED",
        "NOT_FOUND",
        "ALREADY_EXISTS",
        "PERMISSION_DENIED",
        "RESOURCE_EXHAUSTED",
        "FAILED_PRECONDITION",
        "ABORTED",
        "OUT_OF_RANGE",
        "UNIMPLEMENTED",
        "INTERNAL",
        "UNAVAILABLE",
        "DATA_LOSS",
        "UNAUTHENTICATED"
    };

    private final String function;
    private final int code;

    PjrtException(String function, int code, String message) {
        super(function + ": " + codeName(code) + ": " + message);
        this.function = function;
        this.code = code;
    }

    /** The PJRT function that failed, e.g. {@code PJRT_Client_Compile}. */
    public String function() {
        return function;
    }

    /** The {@code PJRT_Error_Code} (absl status code). */
    public int code() {
        return code;
    }

    /** The name of the {@code PJRT_Error_Code}, e.g. {@code INVALID_ARGUMENT}. */
    public String codeName() {
        return codeName(code);
    }

    static String codeName(int code) {
        return code >= 0 && code < CODES.length ? CODES[code] : "code " + code;
    }
}
