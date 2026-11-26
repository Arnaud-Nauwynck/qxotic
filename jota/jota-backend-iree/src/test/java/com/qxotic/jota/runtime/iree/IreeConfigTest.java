package com.qxotic.jota.runtime.iree;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class IreeConfigTest {

    @TempDir Path tmp;

    private IreeConfig config(Properties props, Map<String, String> env) {
        return IreeConfig.from(JotaConfig.of(props, env, null, null));
    }

    @Test
    void defaultsTargetHostCpu() {
        var config = config(new Properties(), Map.of());
        assertTrue(config.compilerLibrary().isEmpty());
        assertEquals(
                List.of(
                        "--iree-hal-target-device=local",
                        "--iree-hal-local-target-device-backends=llvm-cpu",
                        "--iree-llvmcpu-target-cpu=host",
                        "--iree-llvmcpu-link-embedded=false",
                        "--iree-input-demote-f64-to-f32=false",
                        "--iree-input-demote-i64-to-i32=false"),
                config.compilerFlags());
    }

    @Test
    void vmvxHasNoCpuFlag() {
        var props = new Properties();
        props.setProperty(IreeConfig.TARGET_BACKENDS, "vmvx");
        assertEquals(
                List.of(
                        "--iree-hal-target-device=local",
                        "--iree-hal-local-target-device-backends=vmvx",
                        "--iree-input-demote-f64-to-f32=false",
                        "--iree-input-demote-i64-to-i32=false"),
                config(props, Map.of()).compilerFlags());
    }

    @Test
    void gpuDeviceAndExtraFlagsFromEnvironment() {
        var config =
                config(
                        new Properties(),
                        Map.of(
                                "JOTA_IREE_TARGET_DEVICE", "cuda",
                                "JOTA_IREE_COMPILER_FLAGS",
                                        "--iree-cuda-target=sm_80 \"--a=b c\""));
        assertEquals(
                List.of(
                        "--iree-hal-target-device=cuda",
                        "--iree-input-demote-f64-to-f32=false",
                        "--iree-input-demote-i64-to-i32=false",
                        "--iree-cuda-target=sm_80",
                        "--a=b c"),
                config.compilerFlags());
    }

    @Test
    void splitFlags() {
        assertEquals(List.of(), IreeConfig.splitFlags("   "));
        assertEquals(List.of("a", "b"), IreeConfig.splitFlags(" a  b "));
        assertEquals(List.of("x y", "z\"w", ""), IreeConfig.splitFlags("\"x y\" z\\\"w \"\""));
        assertThrows(IllegalArgumentException.class, () -> IreeConfig.splitFlags("\"open"));
    }

    @Test
    void libraryDirectoryIsSearched() throws IOException {
        Path lib = Files.createFile(tmp.resolve("libIREECompiler.so"));
        var props = new Properties();
        props.setProperty(IreeConfig.COMPILER_LIB, tmp.toString());
        assertEquals(lib, config(props, Map.of()).compilerLibrary().orElseThrow());
    }
}
