package com.qxotic.jota.runtime.xla;

import java.io.IOException;
import java.io.Reader;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Properties;

/**
 * Settings of the XLA backend, each looked up by its property key in this order:
 *
 * <ol>
 *   <li>a JVM system property ({@code -Djota.xla.plugin=...});
 *   <li>an environment variable, the key upper-cased with dots as underscores ({@code
 *       JOTA_XLA_PLUGIN});
 *   <li>{@code ./config/jota.properties}, relative to the working directory;
 *   <li>{@code ~/jota/jota.properties}.
 * </ol>
 *
 * <p>The first source that defines a non-blank value wins. Missing or unreadable files are skipped,
 * a malformed one is an error. Values read from a file may use {@code ~/} for the home directory.
 */
public final class XlaConfig {

    /**
     * Path of the PJRT plugin library ({@code xla_cpu_pjrt.so}, {@code pjrt_c_api_cpu_plugin.so},
     * ...).
     */
    public static final String PLUGIN = "jota.xla.plugin";

    /**
     * How programs are handed to the plugin: {@code portable} (VHLO artifact for the plugin's
     * StableHLO version, the default), {@code bytecode} (plain StableHLO bytecode) or {@code text}.
     */
    public static final String FORMAT = "jota.xla.format";

    /** Directory where every program sent to the plugin is also written, for debugging. */
    public static final String DUMP_DIR = "jota.xla.dump";

    /** Maximum number of compiled executables kept per device (least recently used evicted). */
    public static final String CACHE_SIZE = "jota.xla.cache.size";

    /**
     * Whether compiled executables are also kept on disk ({@code true} by default), under the jota
     * kernel cache ({@code jota.cache.root}), in a directory per plugin build.
     */
    public static final String DISK_CACHE = "jota.xla.disk.cache";

    /** Where a value came from. */
    public enum Origin {
        SYSTEM_PROPERTY,
        ENVIRONMENT,
        FILE
    }

    /** A resolved value and its origin ({@code file} is set for {@link Origin#FILE}). */
    public record Value(String key, String value, Origin origin, Path file) {
        public Value {
            Objects.requireNonNull(key, "key");
            Objects.requireNonNull(value, "value");
            Objects.requireNonNull(origin, "origin");
        }

        /** "system property jota.xla.plugin", "environment JOTA_XLA_PLUGIN", or the file path. */
        public String describeOrigin() {
            return switch (origin) {
                case SYSTEM_PROPERTY -> "system property " + key;
                case ENVIRONMENT -> "environment variable " + envName(key);
                case FILE -> key + " in " + file;
            };
        }
    }

    private final Map<String, String> systemProperties;
    private final Map<String, String> environment;
    private final List<Path> files;
    private final Path home;
    private List<Loaded> loaded;

    private record Loaded(Path file, Properties properties) {}

    /**
     * A configuration over explicit sources (for tests): system properties, environment and the
     * files tried in order; {@code home} expands {@code ~/} in file values.
     */
    public XlaConfig(
            Map<String, String> systemProperties,
            Map<String, String> environment,
            List<Path> files,
            Path home) {
        this.systemProperties = Map.copyOf(systemProperties);
        this.environment = Map.copyOf(environment);
        this.files = List.copyOf(files);
        this.home = home;
    }

    /** The configuration of this process: system properties, environment, default files. */
    public static XlaConfig fromEnvironment() {
        Map<String, String> props = new java.util.HashMap<>();
        System.getProperties().forEach((k, v) -> props.put(k.toString(), v.toString()));
        return new XlaConfig(props, System.getenv(), defaultFiles(), userHome());
    }

    /** {@code ./config/jota.properties}, then {@code ~/jota/jota.properties}. */
    public static List<Path> defaultFiles() {
        List<Path> files = new ArrayList<>();
        files.add(Path.of(System.getProperty("user.dir", ".")).resolve("config/jota.properties"));
        Path home = userHome();
        if (home != null) {
            files.add(home.resolve("jota/jota.properties"));
        }
        return List.copyOf(files);
    }

    /** {@code jota.xla.plugin} becomes {@code JOTA_XLA_PLUGIN}. */
    public static String envName(String key) {
        return key.toUpperCase(Locale.ROOT).replace('.', '_').replace('-', '_');
    }

    /** The files this configuration reads, in lookup order. */
    public List<Path> files() {
        return files;
    }

    /** The value of {@code key} with its origin, or empty when no source defines it. */
    public Optional<Value> lookup(String key) {
        String v = systemProperties.get(key);
        if (isSet(v)) {
            return Optional.of(new Value(key, v.strip(), Origin.SYSTEM_PROPERTY, null));
        }
        v = environment.get(envName(key));
        if (isSet(v)) {
            return Optional.of(new Value(key, v.strip(), Origin.ENVIRONMENT, null));
        }
        for (Loaded file : loadedFiles()) {
            v = file.properties().getProperty(key);
            if (isSet(v)) {
                return Optional.of(new Value(key, expandHome(v.strip()), Origin.FILE, file.file()));
            }
        }
        return Optional.empty();
    }

    /** The value of {@code key}, or empty. */
    public Optional<String> get(String key) {
        return lookup(key).map(Value::value);
    }

    /** The value of {@code key}, or {@code defaultValue}. */
    public String get(String key, String defaultValue) {
        return get(key).orElse(defaultValue);
    }

    /** A human-readable list of where {@code key} is looked up, for error hints. */
    public String describeLookup(String key) {
        StringBuilder sb = new StringBuilder();
        sb.append("-D").append(key).append("=..., environment variable ").append(envName(key));
        for (Path file : files) {
            sb.append(", ").append(key).append(" in ").append(file);
        }
        return sb.toString();
    }

    private synchronized List<Loaded> loadedFiles() {
        if (loaded == null) {
            List<Loaded> list = new ArrayList<>();
            for (Path file : files) {
                if (!Files.isRegularFile(file) || !Files.isReadable(file)) {
                    continue;
                }
                Properties p = new Properties();
                try (Reader r = Files.newBufferedReader(file, StandardCharsets.UTF_8)) {
                    p.load(r);
                } catch (IOException e) {
                    throw new UncheckedIOException("Cannot read " + file, e);
                } catch (IllegalArgumentException e) {
                    throw new IllegalArgumentException("Malformed properties file " + file, e);
                }
                list.add(new Loaded(file, p));
            }
            loaded = List.copyOf(list);
        }
        return loaded;
    }

    private String expandHome(String v) {
        if (home != null && (v.equals("~") || v.startsWith("~/"))) {
            return home.resolve(v.substring(Math.min(2, v.length()))).toString();
        }
        return v;
    }

    private static boolean isSet(String v) {
        return v != null && !v.isBlank();
    }

    private static Path userHome() {
        String h = System.getProperty("user.home");
        return h == null || h.isBlank() ? null : Path.of(h);
    }
}
