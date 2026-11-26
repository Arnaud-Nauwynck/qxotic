package com.qxotic.jota.tensor;

import com.qxotic.jota.ir.tir.TIRGraph;

/** Test-only bridge into jota's package-private tracing machinery. */
public final class TracedGraphs {

    private TracedGraphs() {}

    /** The TIR graph recorded by {@link Tracer} for a traced tensor. */
    public static TIRGraph graphOf(Tensor traced) {
        LazyComputation computation =
                InternalTensorAccess.computation(traced)
                        .orElseThrow(() -> new IllegalArgumentException("not a traced tensor"));
        if (computation instanceof IRComputation ir) {
            return ir.graph();
        }
        throw new IllegalArgumentException("not an IR computation: " + computation);
    }
}
