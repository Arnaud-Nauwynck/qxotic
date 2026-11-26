package com.qxotic.jota.runtime.iree.compiler;

import java.util.List;

/** A failure to load the IREE compiler or to compile a program, with its diagnostics. */
public class IreeCompilerException extends RuntimeException {

    private final List<String> diagnostics;

    public IreeCompilerException(String message) {
        this(message, List.of(), null);
    }

    public IreeCompilerException(String message, Throwable cause) {
        this(message, List.of(), cause);
    }

    public IreeCompilerException(String message, List<String> diagnostics) {
        this(message, diagnostics, null);
    }

    public IreeCompilerException(String message, List<String> diagnostics, Throwable cause) {
        super(withDiagnostics(message, diagnostics), cause);
        this.diagnostics = List.copyOf(diagnostics);
    }

    /** Compiler diagnostics collected before the failure, e.g. {@code error: ...}. */
    public List<String> diagnostics() {
        return diagnostics;
    }

    private static String withDiagnostics(String message, List<String> diagnostics) {
        if (diagnostics.isEmpty()) {
            return message;
        }
        return message + "\n" + String.join("\n", diagnostics);
    }
}
