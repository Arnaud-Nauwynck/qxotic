# jota-backend-isolated

Run risky native code — backends such as XLA or IREE, model loading, LLM calls — in a **child JVM**, so that a native crash fails the calls instead of killing your JVM.

A SIGSEGV, `abort()`, native OOM or driver hang inside native code cannot be caught by the JVM.
This module moves that code to a separate process, talks to it over a private Unix domain socket, and turns a child crash into an `IsolatedCrashException` that carries the exit code, the child's stderr tail and the HotSpot fatal error log (`hs_err_pid*.log`).

Pure Java, no native code, no dependency on the rest of jota yet. Design and plan: `_arnaud/TODO-backend-jvm-isolation.md`.

## Use

```java
try (IsolatedJvm child = IsolatedJvm.start(IsolationConfig.defaults())) {
    byte[] reply = child.invoke("llm", "generate", request);          // blocking
    child.callStreaming("llm", "generate", request, token -> show(token)) // streamed
         .get();
} catch (IsolatedCrashException e) {
    log.warn(e.report().summary());   // exit code, signal, hs_err header, stderr tail
}
```

`IsolatedSupervisor` keeps one child available, restarts it after a crash and replays state through `onStart` hooks (for example, reloading the model):

```java
var supervisor = new IsolatedSupervisor(config, 3, Duration.ofMinutes(5));
supervisor.onStart(jvm -> jvm.invoke("llm", "load", modelPath));
supervisor.invoke("llm", "generate", request);
```

## Writing a service (child side)

Implement `com.qxotic.jota.runtime.isolated.spi.IsolatedService` and list it in `META-INF/services/com.qxotic.jota.runtime.isolated.spi.IsolatedService` on the child's classpath (by default the child uses the parent's classpath).
Payloads are opaque bytes; `Wire` writes and reads typed fields.
`CallContext.emit` streams items, `CallContext.isCancelled` reports a cancel or timeout from the parent.
Pass bulk data (tensors, logits) through a `SharedMemoryFile` (in `/dev/shm` on Linux) instead of inline in frames.

The built-in `system` service answers `ping`, `echo`, `pid`, `property`, `sleep`, `stream`, `fail`; its crash methods (`halt`, `segv`, `abort`) only work in a child started with `-Djota.isolated.testCrashes=true`.

## Behaviour

- The parent binds the socket in a private (`rwx------`) temp directory; the child connects and proves itself with a random token passed in its environment.
- One worker thread by default (`serverThreads`): native libraries often keep per-thread state.
- Cancelling is a flag, never a thread interrupt: interrupting native code is unsafe.
- A call timeout cancels the call; with `timeoutAction(KILL)` it kills the child instead, since a native call that hangs never sees the flag.
- `heartbeat(interval, timeout)` pings the child's I/O thread and kills a child that stops answering (wedged but alive); long calls do not trip it.
- The child halts as soon as the parent's socket closes or the parent process exits.
- A clean `close()` deletes the temp work directory unless it holds an `hs_err` file.

## Test

```bash
cd jota && mvn -pl jota-backend-isolated test
```

The tests crash real children (halt, SIGKILL, a real SIGSEGV and a libc `abort()` through FFM, SIGSTOP for the heartbeat, SIGKILL of a parent JVM) and check that the test JVM survives and no child is left behind.
