package com.qxotic.jota.runtime.iree;

import com.qxotic.jota.stablehlo.portable.StableHloVersion;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * IREE backend settings, read through {@link JotaConfig}: each key can be set with {@code -Dkey},
 * the matching {@code JOTA_*} environment variable, {@code ./config/jota.properties} or {@code
 * ~/jota/jota.properties}, in that order of precedence.
 */
public record IreeConfig(
        Optional<JotaConfig.Value> compilerLib,
        String targetDevice,
        String targetBackends,
        String targetCpu,
        List<String> extraFlags,
        Optional<Path> dumpDir,
        Path resolvedCompilerLib,
        Path resolvedRuntimeLib,
        String driver,
        IreeGraphExecutor.Format format,
        StableHloVersion stableHloVersion) {

    /**
     * Path to {@code libIREECompiler.so} / {@code .dylib} / {@code IREECompiler.dll}, or its
     * directory.
     */
    public static final String COMPILER_LIB = "jota.iree.compiler.lib";

    /**
     * HAL target device: {@code local} (default), {@code cuda}, {@code hip}, {@code vulkan}, {@code
     * metal}.
     */
    public static final String TARGET_DEVICE = "jota.iree.target.device";

    /** Backends of the {@code local} device: {@code llvm-cpu} (default) or {@code vmvx}. */
    public static final String TARGET_BACKENDS = "jota.iree.target.backends";

    /** {@code --iree-llvmcpu-target-cpu} for {@code llvm-cpu}; {@code host} by default. */
    public static final String TARGET_CPU = "jota.iree.target.cpu";

    /** Extra compiler flags, space separated, double quotes group a flag that holds spaces. */
    public static final String COMPILER_FLAGS = "jota.iree.compiler.flags";

    /** Directory where every program sent to the compiler, and its output, is dumped. */
    public static final String DUMP_DIR = "jota.iree.dump";

    /**
     * Path to {@code libjota_iree_rt.so} (the IREE runtime built by {@code src/main/native}), or
     * its directory.
     */
    public static final String RUNTIME_LIB = "jota.iree.runtime.lib";

    /**
     * HAL driver / device URI the runtime runs modules on: {@code local-task} (default), {@code
     * local-sync}, {@code cuda}, {@code hip}, ...; must match {@link #TARGET_DEVICE}.
     */
    public static final String DRIVER = "jota.iree.driver";

    /** Program encoding sent to the compiler: {@code text} (default) or {@code portable}. */
    public static final String FORMAT = "jota.iree.format";

    /**
     * StableHLO version a {@code portable} program is written for; default {@code 1.0.0}, which any
     * IREE release reads (IREE upgrades VHLO to its own StableHLO).
     */
    public static final String STABLEHLO_VERSION = "jota.iree.stablehlo.version";

    static final String DEFAULT_DRIVER = "local-task";
    static final String DEFAULT_TARGET_DEVICE = "local";
    static final String DEFAULT_TARGET_BACKENDS = "llvm-cpu";
    static final String DEFAULT_TARGET_CPU = "host";

    public static IreeConfig load() {
        return from(JotaConfig.global());
    }

    public static IreeConfig from(JotaConfig config) {
        Optional<JotaConfig.Value> lib = config.lookup(COMPILER_LIB);
        Path resolved =
                config.getPath(COMPILER_LIB)
                        .map(p -> resolveLibrary(p, COMPILER_LIBRARY_NAMES))
                        .orElse(null);
        return new IreeConfig(
                lib,
                config.get(TARGET_DEVICE, DEFAULT_TARGET_DEVICE),
                config.get(TARGET_BACKENDS, DEFAULT_TARGET_BACKENDS),
                config.get(TARGET_CPU, DEFAULT_TARGET_CPU),
                config.get(COMPILER_FLAGS).map(IreeConfig::splitFlags).orElse(List.of()),
                config.getPath(DUMP_DIR),
                resolved,
                config.getPath(RUNTIME_LIB)
                        .map(p -> resolveLibrary(p, RUNTIME_LIBRARY_NAMES))
                        .orElse(null),
                config.get(DRIVER, DEFAULT_DRIVER),
                config.get(FORMAT)
                        .map(IreeGraphExecutor.Format::parse)
                        .orElse(IreeGraphExecutor.Format.TEXT),
                config.get(STABLEHLO_VERSION)
                        .map(StableHloVersion::parse)
                        .orElse(new StableHloVersion(1, 0, 0)));
    }

    /** The compiler library to load, if configured. */
    public Optional<Path> compilerLibrary() {
        return Optional.ofNullable(resolvedCompilerLib);
    }

    /** The runtime library to load, if configured. */
    public Optional<Path> runtimeLibrary() {
        if (resolvedRuntimeLib != null) {
            return Optional.of(resolvedRuntimeLib);
        }
        // not configured: the copy packaged in this jar (-Piree-native), if any
        return PackagedLibrary.find("jota_iree_rt");
    }

    /** Compiler flags selecting the target, followed by the user's extra flags. */
    public List<String> compilerFlags() {
        var flags = new ArrayList<String>();
        flags.add("--iree-hal-target-device=" + targetDevice);
        if (targetDevice.equals("local")) {
            flags.add("--iree-hal-local-target-device-backends=" + targetBackends);
            if (targetBackends.contains("llvm-cpu")) {
                flags.add("--iree-llvmcpu-target-cpu=" + targetCpu);
                // a system shared library instead of IREE's libc-free embedded ELF: f64 math
                // (cos, log, ...) has no built-in approximation and must resolve from libm
                flags.add("--iree-llvmcpu-link-embedded=false");
            }
        }
        // jota data types are part of the program's meaning: IREE would otherwise demote f64 to
        // f32 (on by default). Extra flags come last, so a user can still opt in.
        flags.add("--iree-input-demote-f64-to-f32=false");
        flags.add("--iree-input-demote-i64-to-i32=false");
        flags.addAll(extraFlags);
        return List.copyOf(flags);
    }

    /** Hint shown when no runtime library is configured or found. */
    public static String runtimeInstallHint() {
        return "build jota-backend-iree/src/main/native (cmake -DIREE_SOURCE_DIR=<iree checkout at"
                + " the compiler's revision>), then set "
                + RUNTIME_LIB
                + " (-D, env "
                + JotaConfig.envName(RUNTIME_LIB)
                + ", ./config/jota.properties or ~/jota/jota.properties) to libjota_iree_rt.so";
    }

    /** Hint shown when no compiler library is configured or found. */
    public static String installHint() {
        return "pip install iree-base-compiler, then set "
                + COMPILER_LIB
                + " (-D"
                + COMPILER_LIB
                + ", env "
                + JotaConfig.envName(COMPILER_LIB)
                + ", ./config/jota.properties or ~/jota/jota.properties) to"
                + " <site-packages>/iree/compiler/_mlir_libs/libIREECompiler.so";
    }

    static final List<String> COMPILER_LIBRARY_NAMES =
            List.of("libIREECompiler.so", "libIREECompiler.dylib", "IREECompiler.dll");
    static final List<String> RUNTIME_LIBRARY_NAMES =
            List.of("libjota_iree_rt.so", "libjota_iree_rt.dylib", "jota_iree_rt.dll");

    /** A directory is searched for the platform's library file name; a file is kept as is. */
    static Path resolveLibrary(Path configured, List<String> names) {
        if (Files.isDirectory(configured)) {
            for (String name : names) {
                Path candidate = configured.resolve(name);
                if (Files.isRegularFile(candidate)) {
                    return candidate;
                }
            }
        }
        return configured;
    }

    /** Splits on whitespace; double quotes group, a backslash escapes the next character. */
    static List<String> splitFlags(String raw) {
        var out = new ArrayList<String>();
        var current = new StringBuilder();
        boolean inQuotes = false;
        boolean any = false;
        for (int i = 0; i < raw.length(); i++) {
            char c = raw.charAt(i);
            if (c == '\\' && i + 1 < raw.length()) {
                current.append(raw.charAt(++i));
                any = true;
            } else if (c == '"') {
                inQuotes = !inQuotes;
                any = true;
            } else if (Character.isWhitespace(c) && !inQuotes) {
                if (any) {
                    out.add(current.toString());
                    current.setLength(0);
                    any = false;
                }
            } else {
                current.append(c);
                any = true;
            }
        }
        if (inQuotes) {
            throw new IllegalArgumentException(
                    "Unbalanced quote in " + COMPILER_FLAGS + ": " + raw);
        }
        if (any) {
            out.add(current.toString());
        }
        return List.copyOf(out);
    }
}
