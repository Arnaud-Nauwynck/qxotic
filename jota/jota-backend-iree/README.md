# jota-backend-iree (experimental)

IREE as a jota backend compiler: TIR graphs exported to StableHLO by `jota-stablehlo`, compiled ahead of time into IREE `.vmfb` modules by the in-process IREE compiler (`libIREECompiler`, stable embedding C API), and run by the IREE C runtime (`libjota_iree_rt`, built from IREE sources by `src/main/native`), both bound with FFM.
Plan and remaining work: `_arnaud/TODO-adapter-iree.md`.

Status (phase A of the plan, host memory):

- `iree` device for jota (`IreeRuntimeProvider`, ServiceLoader): `Environment` can run on it like on any backend; each scheduled kernel step is exported, compiled and run by IREE.
  The shared jota-tensor / jota-memory suites pass on it (`-Piree`: 776 tests, 0 failures, 41 skipped by the testkit's own "panama-only" assumptions).
- `IreeGraphExecutor.execute(TIRGraph, inputs, outputDomain)`: export → compile (cached in memory and as `.vmfb` files on disk) → load → copy inputs in (strided views gathered to row-major) → call `module.main` → outputs as row-major host views;
- `IreeCompiler`: StableHLO text, MLIR bytecode or VHLO in, `.vmfb` bytes out, with diagnostics;
- `IreeDevice` / `IreeModule`: runtime instance + HAL device (`local-task`, `local-sync`), module sessions, calls on dense host tensors.

Not there yet: device-resident memory (every step copies in and out), whole-graph compilation (jota's scheduler hands over one kernel step at a time), GPU drivers.

## Getting the compiler

```bash
uv venv iree-venv && VIRTUAL_ENV=$PWD/iree-venv uv pip install iree-base-compiler
python -c 'import iree.compiler.version as v; print(v.REVISIONS)'   # the IREE revision to build the runtime from
# -> iree-venv/lib/python3.*/site-packages/iree/compiler/_mlir_libs/libIREECompiler.so
```

## Building the runtime library

IREE ships its runtime as static libraries only; `src/main/native` links it into one shared object (runtime only: C, no LLVM, a few minutes).
Build it from the **same IREE revision as the compiler** (`ireeCompilerGetRevision()`, printed by `IreeCompilerTest`), with the `third_party/flatcc` submodule:

```bash
git clone https://github.com/iree-org/iree && git -C iree checkout <compiler revision>
git -C iree submodule update --init --depth 1 third_party/flatcc
```

With Maven, which also packages the library in the jar under `META-INF/native/<os>/<arch>/` (it is then found without any setting, extracted once to the jota cache):

```bash
cd jota && mvnd -pl jota-backend-iree -Piree-native -Diree.source.dir=/path/to/iree install
```

Or by hand, then point `jota.iree.runtime.lib` at the result:

```bash
cmake -S jota-backend-iree/src/main/native -B build-iree-rt -DCMAKE_BUILD_TYPE=Release -DIREE_SOURCE_DIR=$PWD/iree
cmake --build build-iree-rt --target jota_iree_rt -j
# -> build-iree-rt/libjota_iree_rt.so  (CPU drivers local-sync and local-task; -DJOTA_IREE_DRIVERS_GPU=ON for GPU ones)
```

## Configuration

Every key is looked up, first match wins, in:

1. the system property, `-Djota.iree.compiler.lib=...`;
2. the environment variable, the key in upper case with `_` for `.`: `JOTA_IREE_COMPILER_LIB`;
3. `./config/jota.properties`, relative to the working directory (under Maven tests: the module directory);
4. `~/jota/jota.properties`.

Blank values count as unset.
In a properties file a relative path resolves against the file's directory; `~/` expands to the user home everywhere.

| key | default | meaning |
|---|---|---|
| `jota.iree.compiler.lib` | none | `libIREECompiler.so` / `.dylib` / `IREECompiler.dll`, or the directory holding it |
| `jota.iree.runtime.lib` | packaged copy | `libjota_iree_rt.so` (see above), or the directory holding it; unset: the one packaged by `-Piree-native`, if any |
| `jota.iree.driver` | `local-task` | HAL driver / device URI modules run on: `local-task`, `local-sync`, ... |
| `jota.iree.target.device` | `local` | `--iree-hal-target-device`: `local`, `cuda`, `hip`, `vulkan`, `metal` |
| `jota.iree.target.backends` | `llvm-cpu` | backends of the `local` device: `llvm-cpu` or `vmvx` |
| `jota.iree.target.cpu` | `host` | `--iree-llvmcpu-target-cpu` for `llvm-cpu` |
| `jota.iree.compiler.flags` | none | extra compiler flags, space separated, `"..."` to group; they come last and win |
| `jota.iree.format` | `text` | program sent to the compiler: `text` (MLIR) or `portable` (VHLO artifact) |
| `jota.iree.stablehlo.version` | `1.0.0` | StableHLO version of `portable` programs |
| `jota.iree.dump` | none | directory receiving every program sent to the compiler and its `.vmfb` |

Compiled modules are cached under `<KernelCachePaths.versionRoot()>/iree/<hash>.vmfb`; the key covers the program, the compiler flags and the compiler revision.

`JotaConfig` (the lookup) is generic and does not depend on IREE: `jota-backend-xla` has the same lookup as `XlaConfig`; both can move to one class in `jota-tensor`.

## Things worth knowing

- **Signal handlers.** LLVM inside `libIREECompiler` installs its own SIGSEGV / SIGBUS / SIGILL / ... handlers during `llvm-cpu` compiles, replacing the JVM's, and the JVM uses SIGSEGV for ordinary work: the next implicit null check then kills the JVM ("SIGSEGV in libc `syscall`", really LLVM re-raising with `tgkill`).
  `SignalHandlerGuard` captures the JVM's handlers before the compiler loads and restores them after every compiler call.
  For full safety (a JVM fault on another thread during the first `llvm-cpu` compile), preload the JVM's signal chaining: `LD_PRELOAD=$JAVA_HOME/lib/libjsig.so`.
- **f64 math on CPU.** IREE's default `llvm-cpu` output is a libc-free embedded ELF: f64 `cos`, `log`, ... have no built-in approximation and fail to link (`undefined symbol: cos`). The configured flags add `--iree-llvmcpu-link-embedded=false` (a system shared library, resolving them from the process's libm).
- **IREE bug: `reduce(convert(splat constant))`** returns one element instead of the sum (3.14, any target). `TirToStableHlo` folds the `convert` of a splat constant into a constant, so IREE never sees the pattern; details and reproducer in `_arnaud/DRAFT-iree-bug-reduce-convert-splat.md`.
- **f64 / i64.** IREE demotes f64 to f32 by default; the configured flags turn demotion of f64 and i64 off, since jota's data types are part of the program's meaning.
- **Proactor pool.** Since IREE 3.14, HAL devices need an async proactor pool in their create params; `jota_iree_rt_device_create` creates one, as `iree_runtime_instance_try_create_default_device` does.

## Tests

```bash
cd jota && mvnd -pl jota-backend-iree test           # module tests; compiler / runtime ones skip without both libraries
JOTA_IREE_COMPILER_LIB=/path/to/_mlir_libs JOTA_IREE_RUNTIME_LIB=/path/to/build-iree-rt mvnd -pl jota-backend-iree test
JOTA_IREE_COMPILER_LIB=... JOTA_IREE_RUNTIME_LIB=... mvnd -pl jota-backend-iree -Piree test   # + the shared suites on the iree device (~10 min, cold cache)
```
