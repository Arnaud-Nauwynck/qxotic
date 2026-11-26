package com.qxotic.jota.runtime.xla.pjrt;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BOOLEAN;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * A loaded PJRT plugin: the {@code PJRT_Api} function table returned by its {@code GetPjrtApi}.
 *
 * <p>Functions are looked up by name in the table order of {@code pjrt_c_api.h} (append-only
 * upstream), so a plugin built against a newer header works as long as its major version is 0; a
 * function the plugin's table is too short for raises {@link UnsupportedOperationException}.
 *
 * <p>A plugin library is loaded once per process and never unloaded.
 */
public final class PjrtApi {

    /** The {@code PJRT_API_MAJOR} these bindings follow. */
    public static final int MAJOR = 0;

    /** The {@code PJRT_API_MINOR} of the header whose struct layouts these bindings use. */
    public static final int MINOR = 81;

    private static final Linker LINKER = Linker.nativeLinker();
    private static final FunctionDescriptor RETURNS_ERROR = FunctionDescriptor.of(ADDRESS, ADDRESS);
    private static final FunctionDescriptor RETURNS_VOID = FunctionDescriptor.ofVoid(ADDRESS);
    private static final Set<String> VOID_FUNCTIONS =
            Set.of("PJRT_Error_Destroy", "PJRT_Error_Message");

    /** Offset of the first function pointer: struct_size, extension_start, PJRT_Api_Version. */
    private static final long FIRST_FUNCTION = 16 + 24;

    private static final Map<String, Integer> FUNCTION_INDEX = loadFunctionIndex();
    private static final Map<Path, PjrtApi> LOADED = new HashMap<>();

    private final Path library;
    private final MemorySegment api;
    private final long structSize;
    private final int major;
    private final int minor;
    private final MethodHandle[] handles = new MethodHandle[FUNCTION_INDEX.size()];
    private boolean initialized;
    private Map<String, Object> attributes;

    private PjrtApi(Path library) {
        this.library = library;
        SymbolLookup lookup = SymbolLookup.libraryLookup(library, Arena.global());
        MemorySegment getApi =
                lookup.find("GetPjrtApi")
                        .orElseThrow(
                                () ->
                                        new IllegalArgumentException(
                                                library + " is not a PJRT plugin: no GetPjrtApi"));
        MemorySegment raw;
        try {
            raw =
                    (MemorySegment)
                            LINKER.downcallHandle(getApi, FunctionDescriptor.of(ADDRESS))
                                    .invokeExact();
        } catch (Throwable t) {
            throw new IllegalStateException("GetPjrtApi failed in " + library, t);
        }
        if (raw.equals(MemorySegment.NULL)) {
            throw new IllegalStateException("GetPjrtApi returned NULL in " + library);
        }
        this.structSize = raw.reinterpret(8).get(JAVA_LONG, 0);
        if (structSize < FIRST_FUNCTION) {
            throw new IllegalStateException(
                    "PJRT_Api of " + library + " is too small: " + structSize);
        }
        this.api = raw.reinterpret(structSize);
        this.major = api.get(JAVA_INT, 32);
        this.minor = api.get(JAVA_INT, 36);
        if (major != MAJOR) {
            throw new IllegalStateException(
                    "PJRT C API "
                            + version()
                            + " of "
                            + library
                            + " is not supported: these bindings need major version "
                            + MAJOR);
        }
    }

    /** Loads (once per process) the plugin library at {@code library}. */
    public static synchronized PjrtApi load(Path library) {
        Path key = library.toAbsolutePath().normalize();
        if (!Files.isRegularFile(key)) {
            throw new IllegalArgumentException("PJRT plugin not found: " + key);
        }
        return LOADED.computeIfAbsent(key, PjrtApi::new);
    }

    /** The plugin library. */
    public Path library() {
        return library;
    }

    /** The plugin's {@code PJRT_API_MAJOR}. */
    public int majorVersion() {
        return major;
    }

    /** The plugin's {@code PJRT_API_MINOR}. */
    public int minorVersion() {
        return minor;
    }

    /** {@code "major.minor"} of the plugin's C API. */
    public String version() {
        return major + "." + minor;
    }

    /** True when the plugin's table has a non-NULL entry for {@code function}. */
    public boolean supports(String function) {
        Integer index = FUNCTION_INDEX.get(function);
        if (index == null) {
            return false;
        }
        long offset = FIRST_FUNCTION + 8L * index;
        return offset + 8 <= structSize && !api.get(ADDRESS, offset).equals(MemorySegment.NULL);
    }

    /** Calls {@code PJRT_Plugin_Initialize}, once. */
    public synchronized void initialize() {
        if (initialized) {
            return;
        }
        try (Arena arena = Arena.ofConfined()) {
            check("PJRT_Plugin_Initialize", Structs.alloc(arena, 2));
        }
        initialized = true;
    }

    /**
     * {@code PJRT_Plugin_Attributes}: {@code String}, {@code Long}, {@code long[]}, {@code Float}
     * or {@code Boolean} values by name, e.g. {@code stablehlo_current_version} → {@code long[] {1,
     * 9, 3}}.
     */
    public synchronized Map<String, Object> attributes() {
        if (attributes == null) {
            Map<String, Object> map = new LinkedHashMap<>();
            if (supports("PJRT_Plugin_Attributes")) {
                try (Arena arena = Arena.ofConfined()) {
                    MemorySegment args = Structs.alloc(arena, 4);
                    check("PJRT_Plugin_Attributes", args);
                    long n = Structs.u64(args, 3);
                    MemorySegment values = Structs.ptr(args, 2).reinterpret(n * NAMED_VALUE_SIZE);
                    for (long i = 0; i < n; i++) {
                        MemorySegment v = values.asSlice(i * NAMED_VALUE_SIZE, NAMED_VALUE_SIZE);
                        map.put(
                                Structs.string(Structs.ptr(v, 2), Structs.u64(v, 3)),
                                namedValue(v));
                    }
                }
            }
            attributes = Map.copyOf(map);
        }
        return attributes;
    }

    /**
     * The plugin's {@code stablehlo_current_version} attribute as {@code "major.minor.patch"}, or
     * null when the plugin does not report one.
     */
    public String stableHloCurrentVersion() {
        return versionAttribute("stablehlo_current_version");
    }

    /** The plugin's {@code stablehlo_minimum_version} attribute, or null. */
    public String stableHloMinimumVersion() {
        return versionAttribute("stablehlo_minimum_version");
    }

    private String versionAttribute(String name) {
        Object v = attributes().get(name);
        if (v instanceof long[] parts && parts.length == 3) {
            return parts[0] + "." + parts[1] + "." + parts[2];
        }
        return null;
    }

    @Override
    public String toString() {
        return "PJRT " + version() + " (" + library + ")";
    }

    // ------------------------------------------------------------------ calls

    /** Calls {@code function} and throws a {@link PjrtException} for a returned error. */
    void check(String function, MemorySegment args) {
        MemorySegment error = call(function, args);
        if (!error.equals(MemorySegment.NULL)) {
            throw toException(function, error);
        }
    }

    /** Calls {@code function}, returning its {@code PJRT_Error*} (NULL on success). */
    MemorySegment call(String function, MemorySegment args) {
        try {
            return (MemorySegment) handle(function).invokeExact(args);
        } catch (RuntimeException | Error e) {
            throw e;
        } catch (Throwable t) {
            throw new IllegalStateException(function + " failed", t);
        }
    }

    private void callVoid(String function, MemorySegment args) {
        try {
            handle(function).invokeExact(args);
        } catch (RuntimeException | Error e) {
            throw e;
        } catch (Throwable t) {
            throw new IllegalStateException(function + " failed", t);
        }
    }

    /** Calls a {@code void (*fn)(void*)} returned by the plugin, e.g. a deleter (NULL: no-op). */
    void callPointer(MemorySegment fn, MemorySegment arg) {
        if (fn.equals(MemorySegment.NULL)) {
            return;
        }
        try {
            LINKER.downcallHandle(fn, RETURNS_VOID).invokeExact(arg);
        } catch (RuntimeException | Error e) {
            throw e;
        } catch (Throwable t) {
            throw new IllegalStateException("plugin callback failed", t);
        }
    }

    /** Converts and destroys {@code error}. */
    PjrtException toException(String function, MemorySegment error) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment msg = Structs.alloc(arena, 5);
            Structs.ptr(msg, 2, error);
            callVoid("PJRT_Error_Message", msg);
            String message = Structs.string(Structs.ptr(msg, 3), Structs.u64(msg, 4));
            int code = 2; // UNKNOWN
            MemorySegment codeArgs = Structs.allocEndingInInt(arena, 4);
            Structs.ptr(codeArgs, 2, error);
            if (call("PJRT_Error_GetCode", codeArgs).equals(MemorySegment.NULL)) {
                code = Structs.i32(codeArgs, 3);
            }
            return new PjrtException(function, code, message);
        } finally {
            destroyError(error);
        }
    }

    private void destroyError(MemorySegment error) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment args = Structs.alloc(arena, 3);
            Structs.ptr(args, 2, error);
            callVoid("PJRT_Error_Destroy", args);
        }
    }

    /** Waits for {@code event} (NULL is a no-op), then destroys it; throws its error if any. */
    void awaitAndDestroy(String what, MemorySegment event) {
        if (event.equals(MemorySegment.NULL)) {
            return;
        }
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment args = Structs.alloc(arena, 3);
            Structs.ptr(args, 2, event);
            MemorySegment error = call("PJRT_Event_Await", args);
            if (!error.equals(MemorySegment.NULL)) {
                throw toException(what, error);
            }
        } finally {
            destroy("PJRT_Event_Destroy", event);
        }
    }

    /** Calls a {@code PJRT_X_Destroy(args{size, ext, handle})} function. */
    void destroy(String function, MemorySegment handle) {
        if (handle.equals(MemorySegment.NULL)) {
            return;
        }
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment args = Structs.alloc(arena, 3);
            Structs.ptr(args, 2, handle);
            check(function, args);
        }
    }

    private synchronized MethodHandle handle(String function) {
        Integer index = FUNCTION_INDEX.get(function);
        if (index == null) {
            throw new IllegalArgumentException("unknown PJRT function " + function);
        }
        MethodHandle h = handles[index];
        if (h == null) {
            long offset = FIRST_FUNCTION + 8L * index;
            MemorySegment fn =
                    offset + 8 <= structSize ? api.get(ADDRESS, offset) : MemorySegment.NULL;
            if (fn.equals(MemorySegment.NULL)) {
                throw new UnsupportedOperationException(
                        function
                                + " is not provided by the PJRT plugin "
                                + library
                                + " (C API "
                                + version()
                                + ")");
            }
            h =
                    LINKER.downcallHandle(
                            fn, VOID_FUNCTIONS.contains(function) ? RETURNS_VOID : RETURNS_ERROR);
            handles[index] = h;
        }
        return h;
    }

    // ------------------------------------------------------------ named values

    /** sizeof(PJRT_NamedValue): size, ext, name, name_size, type, union, value_size. */
    static final long NAMED_VALUE_SIZE = 7 * Structs.SLOT;

    private static Object namedValue(MemorySegment v) {
        int type = Structs.i32(v, 4);
        long valueSize = Structs.u64(v, 6);
        long unionOffset = 5 * Structs.SLOT;
        return switch (type) {
            case 0 -> Structs.string(Structs.ptr(v, 5), valueSize);
            case 1 -> v.get(JAVA_LONG, unionOffset);
            case 2 -> {
                MemorySegment p = Structs.ptr(v, 5);
                yield valueSize == 0
                        ? new long[0]
                        : p.reinterpret(valueSize * 8).toArray(JAVA_LONG);
            }
            case 3 -> v.get(JAVA_FLOAT, unionOffset);
            case 4 -> v.get(JAVA_BOOLEAN, unionOffset);
            default -> "<PJRT_NamedValue type " + type + ">";
        };
    }

    private static Map<String, Integer> loadFunctionIndex() {
        String resource = "pjrt-api-functions.txt";
        try (InputStream in = PjrtApi.class.getResourceAsStream(resource)) {
            if (in == null) {
                throw new IllegalStateException("missing resource " + resource);
            }
            List<String> names = new ArrayList<>();
            try (BufferedReader r =
                    new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8))) {
                for (String line; (line = r.readLine()) != null; ) {
                    line = line.strip();
                    if (!line.isEmpty() && !line.startsWith("#")) {
                        names.add(line);
                    }
                }
            }
            Map<String, Integer> index = new HashMap<>();
            for (int i = 0; i < names.size(); i++) {
                index.put(names.get(i), i);
            }
            return Map.copyOf(index);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /** For diagnostics: {@code attributes()} rendered with arrays expanded. */
    public static String format(Object value) {
        return value instanceof long[] a ? Arrays.toString(a) : String.valueOf(value);
    }
}
