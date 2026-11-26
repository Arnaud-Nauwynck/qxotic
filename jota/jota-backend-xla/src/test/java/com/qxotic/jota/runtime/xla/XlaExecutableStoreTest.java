package com.qxotic.jota.runtime.xla;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class XlaExecutableStoreTest {

    @TempDir Path dir;

    @Test
    void storesLoadsAndDeletes() {
        XlaExecutableStore store = new XlaExecutableStore(dir, "abc");
        assertTrue(store.load("k").isEmpty());
        store.store("k", new byte[] {1, 2, 3});
        assertArrayEquals(new byte[] {1, 2, 3}, store.load("k").orElseThrow());
        assertEquals(dir.resolve("abc"), store.directory());
        store.store("k", new byte[] {4});
        assertArrayEquals(new byte[] {4}, store.load("k").orElseThrow());
        store.delete("k");
        assertTrue(store.load("k").isEmpty());
        assertTrue(store.isEnabled());
    }

    @Test
    void disabledStoreKeepsNothing() {
        XlaExecutableStore store = XlaExecutableStore.disabled();
        store.store("k", new byte[] {1});
        assertTrue(store.load("k").isEmpty());
        assertFalse(store.isEnabled());
    }

    @Test
    void unwritableDirectoryDisablesTheStore() throws IOException {
        Path file = Files.writeString(dir.resolve("not-a-dir"), "x");
        XlaExecutableStore store = new XlaExecutableStore(file, "abc");
        store.store("k", new byte[] {1});
        assertFalse(store.isEnabled());
    }

    @Test
    void fingerprintDependsOnPluginAndVersions() throws IOException {
        Path a = Files.writeString(dir.resolve("a.so"), "a");
        Path b = Files.writeString(dir.resolve("b.so"), "b");
        String fa = XlaExecutableStore.fingerprint(a, "0.81", "cpu");
        assertEquals(fa, XlaExecutableStore.fingerprint(a, "0.81", "cpu"));
        assertEquals(16, fa.length());
        assertNotEquals(fa, XlaExecutableStore.fingerprint(b, "0.81", "cpu"));
        assertNotEquals(fa, XlaExecutableStore.fingerprint(a, "0.82", "cpu"));
        assertNotEquals(fa, XlaExecutableStore.fingerprint(a, "0.81", "cpu 2"));
    }
}
