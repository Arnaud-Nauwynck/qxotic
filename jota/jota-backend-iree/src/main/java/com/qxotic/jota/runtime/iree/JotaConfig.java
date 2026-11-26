package com.qxotic.jota.runtime.iree;

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
 * Layered configuration lookup for jota settings such as {@code jota.iree.compiler.lib}.
 *
 * <p>A key is looked up, first match wins, in:
 *
 * <ol>
 *   <li>the JVM system property of the same name ({@code -Djota.iree.compiler.lib=...});
 *   <li>the environment variable named by {@link #envName(String)} ({@code
 *       JOTA_IREE_COMPILER_LIB});
 *   <li>the project file {@code ./config/jota.properties}, relative to the working directory;
 *   <li>the user file {@code ~/jota/jota.properties}.
 * </ol>
 *
 * <p>A blank value counts as unset at every level, so an empty {@code -D} (as Maven passes for an
 * unset property) falls through to the next level. Missing files are skipped; an unreadable file is
 * an error.
 *
 * <p>Path values read from a file are resolved against that file's directory when relative, and a
 * leading {@code ~/} expands to the user home at every level (see {@link #getPath(String)}).
 */
public final class JotaConfig {

    /** Project-level file, relative to the working directory. */
    public static final Path PROJECT_FILE = Path.of("config", "jota.properties");

    /** User-level file, relative to the user home. */
    public static final Path USER_FILE = Path.of("jota", "jota.properties");

    /** Where a value came from, in decreasing precedence. */
    public enum Origin {
        SYSTEM_PROPERTY,
        ENVIRONMENT,
        PROJECT_FILE,
        USER_FILE
    }

    /** A resolved value and its origin; {@code file} is set for the two file origins. */
    public record Value(String key, String value, Origin origin, Path file) {
        public Value {
            Objects.requireNonNull(key, "key");
            Objects.requireNonNull(value, "value");
            Objects.requireNonNull(origin, "origin");
        }

        /** Human-readable origin, e.g. {@code env JOTA_IREE_COMPILER_LIB}. */
        public String describeOrigin() {
            return switch (origin) {
                case SYSTEM_PROPERTY -> "-D" + key;
                case ENVIRONMENT -> "env " + envName(key);
                case PROJECT_FILE, USER_FILE -> String.valueOf(file);
            };
        }
    }

    private record FileLayer(Origin origin, Path file, Properties properties) {}

    private static volatile JotaConfig global;

    private final Properties systemProperties;
    private final Map<String, String> environment;
    private final Path userHome;
    private final List<FileLayer> files;

    private JotaConfig(
            Properties systemProperties,
            Map<String, String> environment,
            Path workingDirectory,
            Path userHome) {
        this.systemProperties = systemProperties;
        this.environment = environment;
        this.userHome = userHome;
        var layers = new ArrayList<FileLayer>(2);
        if (workingDirectory != null) {
            loadFile(Origin.PROJECT_FILE, workingDirectory.resolve(PROJECT_FILE), layers);
        }
        if (userHome != null) {
            loadFile(Origin.USER_FILE, userHome.resolve(USER_FILE), layers);
        }
        this.files = List.copyOf(layers);
    }

    /**
     * The process-wide configuration: system properties, environment, {@code
     * ${user.dir}/config/jota.properties}, {@code ${user.home}/jota/jota.properties}. Files are
     * read once, on first use; system properties and the environment are read on every lookup.
     */
    public static JotaConfig global() {
        JotaConfig config = global;
        if (config == null) {
            synchronized (JotaConfig.class) {
                config = global;
                if (config == null) {
                    config = global = fromProcess();
                }
            }
        }
        return config;
    }

    /** Forgets the cached {@link #global()} instance, so the files are read again. */
    public static synchronized void reloadGlobal() {
        global = null;
    }

    private static JotaConfig fromProcess() {
        String userDir = System.getProperty("user.dir");
        String home = System.getProperty("user.home");
        return new JotaConfig(
                System.getProperties(),
                System.getenv(),
                userDir == null || userDir.isBlank() ? null : Path.of(userDir),
                home == null || home.isBlank() ? null : Path.of(home));
    }

    /**
     * An explicit configuration, for tests and embedders; {@code workingDirectory} and {@code
     * userHome} may be null to skip the corresponding file.
     */
    public static JotaConfig of(
            Properties systemProperties,
            Map<String, String> environment,
            Path workingDirectory,
            Path userHome) {
        return new JotaConfig(
                Objects.requireNonNull(systemProperties, "systemProperties"),
                Map.copyOf(Objects.requireNonNull(environment, "environment")),
                workingDirectory,
                userHome);
    }

    /**
     * Environment variable name for a key: upper case, every character outside {@code [A-Za-z0-9]}
     * replaced by {@code _}; {@code jota.iree.compiler.lib} gives {@code JOTA_IREE_COMPILER_LIB}.
     */
    public static String envName(String key) {
        return key.replaceAll("[^A-Za-z0-9]", "_").toUpperCase(Locale.ROOT);
    }

    /** The raw value and where it came from, or empty when no level sets it. */
    public Optional<Value> lookup(String key) {
        Objects.requireNonNull(key, "key");
        String fromProperty = trimToNull(systemProperties.getProperty(key));
        if (fromProperty != null) {
            return Optional.of(new Value(key, fromProperty, Origin.SYSTEM_PROPERTY, null));
        }
        String fromEnv = trimToNull(environment.get(envName(key)));
        if (fromEnv != null) {
            return Optional.of(new Value(key, fromEnv, Origin.ENVIRONMENT, null));
        }
        for (FileLayer layer : files) {
            String fromFile = trimToNull(layer.properties().getProperty(key));
            if (fromFile != null) {
                return Optional.of(new Value(key, fromFile, layer.origin(), layer.file()));
            }
        }
        return Optional.empty();
    }

    public Optional<String> get(String key) {
        return lookup(key).map(Value::value);
    }

    public String get(String key, String defaultValue) {
        return get(key).orElse(defaultValue);
    }

    public boolean getBoolean(String key, boolean defaultValue) {
        return get(key).map(v -> Boolean.parseBoolean(v.trim())).orElse(defaultValue);
    }

    /**
     * A path value: a leading {@code ~/} (or a lone {@code ~}) expands to the user home, and a
     * relative path from a properties file resolves against that file's directory, so {@code
     * ~/jota/jota.properties} can say {@code compiler.lib=lib/libIREECompiler.so}. Relative paths
     * from {@code -D} or the environment stay relative to the working directory.
     */
    public Optional<Path> getPath(String key) {
        return lookup(key).map(this::toPath);
    }

    private Path toPath(Value value) {
        String raw = value.value();
        if (userHome != null && (raw.equals("~") || raw.startsWith("~/"))) {
            return raw.length() == 1 ? userHome : userHome.resolve(raw.substring(2));
        }
        Path path = Path.of(raw);
        if (!path.isAbsolute() && value.file() != null) {
            Path dir = value.file().toAbsolutePath().getParent();
            if (dir != null) {
                return dir.resolve(path).normalize();
            }
        }
        return path;
    }

    /** The properties files that exist and were loaded, in precedence order. */
    public List<Path> loadedFiles() {
        return files.stream().map(FileLayer::file).toList();
    }

    private static void loadFile(Origin origin, Path file, List<FileLayer> out) {
        if (!Files.isRegularFile(file)) {
            return;
        }
        var properties = new Properties();
        try (Reader reader = Files.newBufferedReader(file, StandardCharsets.UTF_8)) {
            properties.load(reader);
        } catch (IOException e) {
            throw new UncheckedIOException("Cannot read jota configuration file " + file, e);
        }
        out.add(new FileLayer(origin, file, properties));
    }

    private static String trimToNull(String value) {
        if (value == null) {
            return null;
        }
        String trimmed = value.trim();
        return trimmed.isEmpty() ? null : trimmed;
    }
}
