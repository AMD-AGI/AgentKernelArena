# Backend-neutral GPU implementation

The task permits the GPU implementation backends installed in its pinned runtime.
Follow candidate.editable and the task README, including any multifile replacement
manifest. Preserve numerical semantics, precision and the fixed workload.

Profiling and workload analysis are recommended, not mandatory. Choose your own
tools to study kernel time, actual shapes, layouts, call frequency and fusion
opportunities. Retain useful diagnostics outside protected task files. Optimize
end-to-end throughput; a faster isolated kernel may have no measurable model gain.
Never change benchmark conditions or treat optional diagnostic runs as acceptance.
