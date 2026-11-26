# jota-backend-xla

Runs jota graphs on XLA: a traced `TIRGraph` is exported to StableHLO by `jota-stablehlo`, then compiled and executed by an XLA **PJRT plugin** through the PJRT C API, called with the FFM API.
No XLA binary is bundled and nothing is built with Bazel: you point jota at a plugin library.

Status: early; validated on the XLA CPU plugin built from openxla/xla (module tests, and the shared jota suite with `-Pxla`: 0 failures).
`XlaRuntimeProvider` registers the `xla` device type: one `xla:<n>` device per addressable device of the plugin.
`XlaComputeEngine` exports each graph to StableHLO, compiles it once (cached by the SHA-256 of the encoded program, least recently used evicted), copies the inputs to the device, executes, and copies the result back.
Tensors of an `xla` device are host memory with an optional device copy (`XlaMemory`): an execution result stays on the device as a PJRT buffer, and the next execution that reads the whole memory (same dims, row-major, offset 0) uses that buffer directly.
The host copy is filled on the first `base()` call, which also drops the device copy since the caller may write; anything else (views with an offset or other dims, memory of other devices) is uploaded for the execution.
There is no kernel service: XLA compiles whole graphs.
Plan: `_arnaud/TODO-adapter-xla.md` at the repository root.

## Plugins

Any PJRT plugin with C API major version 0 works; the bindings follow the struct layouts of `pjrt_c_api.h` 0.81 and newer plugins stay compatible (the API is append-only).

- XLA CPU: `xla_plugins/xla_cpu_pjrt/xla_cpu_pjrt.so` from the `xla-cpu-pjrt` wheel on PyPI (unzip the wheel), or `//xla/pjrt/c:pjrt_c_api_cpu_plugin.so` built from openxla/xla.
- XLA CUDA: `xla_cuda_plugin.so` from the `jax-cuda12-pjrt` / `jax-cuda13-pjrt` wheels.

## Configuration

Each setting is looked up, first match wins, in:

1. a system property, `-Djota.xla.plugin=...`;
2. an environment variable, the key upper-cased with `_` for `.`: `JOTA_XLA_PLUGIN`;
3. `./config/jota.properties`, relative to the working directory;
4. `~/jota/jota.properties`.

| key | meaning | default |
|---|---|---|
| `jota.xla.plugin` | path of the PJRT plugin library | none (backend unavailable) |
| `jota.xla.format` | program encoding: `portable` (VHLO artifact for the plugin's StableHLO version), `bytecode` or `text` | `portable` |
| `jota.xla.dump` | directory receiving every program compiled (`<key>.mlir` text and `<key>.bin` as sent) | none |
| `jota.xla.cache.size` | compiled executables kept per device | `256` |

Values read from a file may start with `~/`.

## Tests

`XlaConfigTest` and the plugin-free parts of `XlaRuntimeTest` (protobuf bytes, the probe without a plugin) always run.
With a plugin configured, `PjrtSmokeTest` prints the plugin's attributes and devices, compiles hand-written StableHLO and runs exported jota graphs in all three formats; `XlaRuntimeTest` runs tensor programs on `xla:0` and compares them with panama.
The `xla` profile also runs the shared jota-tensor / jota-memory suites on the `xla` device.

```bash
mvn -pl jota-backend-xla test -Djota.xla.plugin=/path/to/xla_cpu_pjrt.so
mvn -pl jota-backend-xla test -Pxla -Djota.xla.plugin=/path/to/xla_cpu_pjrt.so
```
