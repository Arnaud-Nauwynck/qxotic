package com.qxotic.jota.runtime.xla;

import com.qxotic.jota.runtime.xla.pjrt.PjrtApi;
import com.qxotic.jota.stablehlo.bytecode.MlirBytecodeWriter;
import com.qxotic.jota.stablehlo.ir.Program;
import com.qxotic.jota.stablehlo.portable.PortableArtifact;
import com.qxotic.jota.stablehlo.portable.StableHloVersion;
import com.qxotic.jota.stablehlo.text.MlirPrinter;
import java.nio.charset.StandardCharsets;
import java.util.Locale;

/** Encodes a StableHLO {@link Program} the way a PJRT plugin's {@code "mlir"} format reads it. */
public final class XlaPrograms {

    /** The PJRT program format all encodings use. */
    public static final String PJRT_FORMAT = "mlir";

    /** How the program is written; see {@link XlaConfig#FORMAT}. */
    public enum Format {
        /** A VHLO portable artifact for the plugin's StableHLO version: what jax sends. */
        PORTABLE,
        /** Plain StableHLO bytecode. */
        BYTECODE,
        /** The MLIR text (pretty form); for debugging. */
        TEXT;

        /** {@code "portable"}, {@code "bytecode"} or {@code "text"}, any case. */
        public static Format parse(String text) {
            try {
                return valueOf(text.strip().toUpperCase(Locale.ROOT));
            } catch (IllegalArgumentException e) {
                throw new IllegalArgumentException(
                        "unknown XLA program format '"
                                + text
                                + "' (expected portable, bytecode or text)");
            }
        }
    }

    private XlaPrograms() {}

    /**
     * The StableHLO version to write for {@code api}: the plugin's {@code
     * stablehlo_current_version}, capped at the newest version jota-stablehlo writes; the latter
     * when the plugin reports none.
     */
    public static StableHloVersion target(PjrtApi api) {
        String reported = api.stableHloCurrentVersion();
        if (reported == null) {
            return StableHloVersion.CURRENT;
        }
        StableHloVersion v = StableHloVersion.parse(reported);
        if (v.compareTo(StableHloVersion.CURRENT) > 0) {
            return StableHloVersion.CURRENT;
        }
        if (!v.isSupported()) {
            throw new IllegalStateException(
                    "PJRT plugin "
                            + api.library()
                            + " reads StableHLO "
                            + v
                            + ", older than the oldest version jota writes ("
                            + StableHloVersion.MINIMUM
                            + ")");
        }
        return v;
    }

    /** {@code program} in {@code format}, for a reader of StableHLO {@code target}. */
    public static byte[] encode(Program program, Format format, StableHloVersion target) {
        return switch (format) {
            case PORTABLE -> PortableArtifact.serialize(program, target);
            case BYTECODE ->
                    MlirBytecodeWriter.writePlain(
                            program, PortableArtifact.bytecodeVersion(target), target, "jota");
            case TEXT -> MlirPrinter.print(program).getBytes(StandardCharsets.UTF_8);
        };
    }
}
