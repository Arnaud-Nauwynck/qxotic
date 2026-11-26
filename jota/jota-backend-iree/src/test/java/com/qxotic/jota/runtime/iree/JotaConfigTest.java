package com.qxotic.jota.runtime.iree;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class JotaConfigTest {

    private static final String KEY = "jota.iree.compiler.lib";

    @TempDir Path tmp;

    private Path work() throws IOException {
        return Files.createDirectories(tmp.resolve("work"));
    }

    private Path home() throws IOException {
        return Files.createDirectories(tmp.resolve("home"));
    }

    private static void write(Path file, String content) throws IOException {
        Files.createDirectories(file.getParent());
        Files.writeString(file, content);
    }

    private static Properties props(String... keyValues) {
        var p = new Properties();
        for (int i = 0; i < keyValues.length; i += 2) {
            p.setProperty(keyValues[i], keyValues[i + 1]);
        }
        return p;
    }

    @Test
    void envNameIsUpperSnakeCase() {
        assertEquals("JOTA_IREE_COMPILER_LIB", JotaConfig.envName("jota.iree.compiler.lib"));
        assertEquals("JOTA_A_B_C", JotaConfig.envName("jota.a-b.c"));
    }

    @Test
    void precedenceIsPropertyThenEnvThenProjectFileThenUserFile() throws IOException {
        Path work = work();
        Path home = home();
        write(work.resolve("config/jota.properties"), KEY + "=/project\n");
        write(home.resolve("jota/jota.properties"), KEY + "=/user\n");
        var env = Map.of("JOTA_IREE_COMPILER_LIB", "/env");

        var all = JotaConfig.of(props(KEY, "/prop"), env, work, home).lookup(KEY).orElseThrow();
        assertEquals("/prop", all.value());
        assertEquals(JotaConfig.Origin.SYSTEM_PROPERTY, all.origin());
        assertEquals("-D" + KEY, all.describeOrigin());

        var noProp = JotaConfig.of(props(), env, work, home).lookup(KEY).orElseThrow();
        assertEquals("/env", noProp.value());
        assertEquals(JotaConfig.Origin.ENVIRONMENT, noProp.origin());
        assertEquals("env JOTA_IREE_COMPILER_LIB", noProp.describeOrigin());

        var filesOnly = JotaConfig.of(props(), Map.of(), work, home).lookup(KEY).orElseThrow();
        assertEquals("/project", filesOnly.value());
        assertEquals(JotaConfig.Origin.PROJECT_FILE, filesOnly.origin());

        var userOnly = JotaConfig.of(props(), Map.of(), null, home).lookup(KEY).orElseThrow();
        assertEquals("/user", userOnly.value());
        assertEquals(JotaConfig.Origin.USER_FILE, userOnly.origin());
    }

    @Test
    void blankValuesFallThrough() throws IOException {
        Path home = home();
        write(home.resolve("jota/jota.properties"), KEY + "=/user\n");
        var config =
                JotaConfig.of(props(KEY, "  "), Map.of("JOTA_IREE_COMPILER_LIB", ""), work(), home);
        assertEquals("/user", config.get(KEY).orElseThrow());
    }

    @Test
    void missingEverywhereIsEmpty() throws IOException {
        var config = JotaConfig.of(props(), Map.of(), work(), home());
        assertTrue(config.lookup(KEY).isEmpty());
        assertEquals("dflt", config.get(KEY, "dflt"));
        assertTrue(config.loadedFiles().isEmpty());
    }

    @Test
    void relativePathFromFileResolvesAgainstThatFile() throws IOException {
        Path home = home();
        write(home.resolve("jota/jota.properties"), KEY + "=lib/libIREECompiler.so\n");
        var config = JotaConfig.of(props(), Map.of(), null, home);
        assertEquals(
                home.resolve("jota/lib/libIREECompiler.so").toAbsolutePath().normalize(),
                config.getPath(KEY).orElseThrow());
        assertEquals(List.of(home.resolve("jota/jota.properties")), config.loadedFiles());
    }

    @Test
    void relativePathFromPropertyStaysRelative() throws IOException {
        var config = JotaConfig.of(props(KEY, "rel/lib.so"), Map.of(), work(), home());
        assertEquals(Path.of("rel/lib.so"), config.getPath(KEY).orElseThrow());
    }

    @Test
    void tildeExpandsToUserHome() throws IOException {
        Path home = home();
        var config =
                JotaConfig.of(
                        props(), Map.of("JOTA_IREE_COMPILER_LIB", "~/iree/lib.so"), work(), home);
        assertEquals(home.resolve("iree/lib.so"), config.getPath(KEY).orElseThrow());
    }

    @Test
    void booleans() throws IOException {
        var config = JotaConfig.of(props("a", "true", "b", "no"), Map.of(), work(), home());
        assertTrue(config.getBoolean("a", false));
        assertEquals(false, config.getBoolean("b", true));
        assertTrue(config.getBoolean("c", true));
    }
}
