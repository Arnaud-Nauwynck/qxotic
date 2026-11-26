package com.qxotic.jota.runtime.iree;

import com.qxotic.jota.Device;
import com.qxotic.jota.ir.tir.TIRGraph;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.runtime.ComputeEngine;
import com.qxotic.jota.tensor.Tensor;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Executes the TIR graphs jota schedules for the {@code iree} device (one per kernel step) through
 * {@link IreeGraphExecutor}: StableHLO export, {@code .vmfb} compile (cached), run, result copied
 * into this device's host memory.
 */
final class IreeComputeEngine implements ComputeEngine {

    private final IreeMemoryDomain memoryDomain;
    private final IreeGraphExecutor executor;

    IreeComputeEngine(IreeMemoryDomain memoryDomain, IreeGraphExecutor executor) {
        this.memoryDomain = Objects.requireNonNull(memoryDomain, "memoryDomain");
        this.executor = Objects.requireNonNull(executor, "executor");
    }

    @Override
    public Device device() {
        return memoryDomain.device();
    }

    /**
     * False for now: the first {@code llvm-cpu} compile is where LLVM installs its signal handlers,
     * and other threads compiling or running Java code at that moment can trip over them (see
     * {@code SignalHandlerGuard}).
     */
    @Override
    public boolean supportsParallelPrecompile() {
        return false;
    }

    @Override
    public void precompile(TIRGraph graph) {
        executor.prepare(graph);
    }

    @Override
    public MemoryView<?> execute(TIRGraph graph, List<Tensor> inputs) {
        if (graph.inputs().size() != inputs.size()) {
            throw new IllegalArgumentException(
                    "Expected " + graph.inputs().size() + " inputs but got " + inputs.size());
        }
        if (graph.outputs().size() != 1) {
            throw new IllegalArgumentException(
                    "Expected a single-output graph, got " + graph.outputs().size() + " outputs");
        }
        var views = new ArrayList<MemoryView<MemorySegment>>(inputs.size());
        for (Tensor t : inputs) {
            views.add(hostView(t.materialize()));
        }
        return executor.execute(graph, views, memoryDomain).getFirst();
    }

    @SuppressWarnings("unchecked")
    private static MemoryView<MemorySegment> hostView(MemoryView<?> view) {
        if (!(view.memory().base() instanceof MemorySegment)) {
            throw new IllegalArgumentException(
                    "IREE backend inputs must be in host memory (MemorySegment), got "
                            + view.memory().base().getClass().getName()
                            + " on "
                            + view.memory().device());
        }
        return (MemoryView<MemorySegment>) view;
    }
}
