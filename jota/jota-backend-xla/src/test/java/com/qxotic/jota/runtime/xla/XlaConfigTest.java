package com.qxotic.jota.runtime.xla;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class XlaConfigTest {

    @TempDir Path dir;

    private Path project;
    private Path user;

    private XlaConfig config(Map<String, String> props, Map<String, String> env)
            throws IOException {
        project = dir.resolve("project/config/jota.properties");
        user = dir.resolve("home/jota/jota.properties");
        return new XlaConfig(props, env, List.of(project, user), dir.resolve("home"));
    }

    private static void write(Path file, String content) throws IOException {
        Files.createDirectories(file.getParent());
        Files.writeString(file, content);
    }

    @Test
    void envName() {
        assertEquals("JOTA_XLA_PLUGIN", XlaConfig.envName("jota.xla.plugin"));
    }

    @Test
    void systemPropertyOverridesEnvironmentAndFiles() throws IOException {
        XlaConfig c =
                config(
                        Map.of("jota.xla.plugin", "/from/property"),
                        Map.of("JOTA_XLA_PLUGIN", "/from/env"));
        write(project, "jota.xla.plugin=/from/project\n");
        XlaConfig.Value v = c.lookup(XlaConfig.PLUGIN).orElseThrow();
        assertEquals("/from/property", v.value());
        assertEquals(XlaConfig.Origin.SYSTEM_PROPERTY, v.origin());
    }

    @Test
    void environmentOverridesFiles() throws IOException {
        XlaConfig c = config(Map.of(), Map.of("JOTA_XLA_PLUGIN", "/from/env"));
        write(project, "jota.xla.plugin=/from/project\n");
        assertEquals("/from/env", c.get(XlaConfig.PLUGIN).orElseThrow());
        assertEquals(XlaConfig.Origin.ENVIRONMENT, c.lookup(XlaConfig.PLUGIN).get().origin());
    }

    @Test
    void projectFileBeforeUserFile() throws IOException {
        XlaConfig c = config(Map.of(), Map.of());
        write(project, "jota.xla.plugin=/from/project\n");
        write(user, "jota.xla.plugin=/from/user\njota.xla.format=text\n");
        XlaConfig.Value v = c.lookup(XlaConfig.PLUGIN).orElseThrow();
        assertEquals("/from/project", v.value());
        assertEquals(project, v.file());
        // a key the project file lacks falls through to the user file
        assertEquals("text", c.get(XlaConfig.FORMAT, "portable"));
    }

    @Test
    void blankValuesAreUnset() throws IOException {
        XlaConfig c = config(Map.of("jota.xla.plugin", " "), Map.of("JOTA_XLA_PLUGIN", ""));
        write(user, "jota.xla.plugin=~/plugins/xla_cpu_pjrt.so\n");
        XlaConfig.Value v = c.lookup(XlaConfig.PLUGIN).orElseThrow();
        assertEquals(XlaConfig.Origin.FILE, v.origin());
        assertEquals(dir.resolve("home/plugins/xla_cpu_pjrt.so").toString(), v.value());
    }

    @Test
    void missingEverywhere() throws IOException {
        XlaConfig c = config(Map.of(), Map.of());
        assertTrue(c.lookup(XlaConfig.PLUGIN).isEmpty());
        assertEquals("portable", c.get(XlaConfig.FORMAT, "portable"));
        String hint = c.describeLookup(XlaConfig.PLUGIN);
        assertTrue(hint.contains("JOTA_XLA_PLUGIN") && hint.contains(user.toString()), hint);
    }
}
