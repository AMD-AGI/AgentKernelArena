# points_in_boxes: native HIP task

Optimize only `src/points_in_boxes_cuda.hip`. All task inputs, references,
CPU comparisons, wrappers, bindings, compiler flags, launch boundaries, and
benchmark helpers are protected. The initial implementation is present and
written in HIP. Arena freezes that initial candidate in a separate workspace
for every baseline action; final candidate actions always build and execute the
submitted implementation, with no baseline fallback.

The complete independent manifest in `workload.json` includes **20 cases**
from the original 5 shapes, including every measured operation/layout.
The original harness remains `scripts/task_runner.py`; its seed schedule,
numerical tolerances and output-contract checks are retained. Extra checks in
`scripts/reference_checks.py` provide small independent reference controls and
cover measured variants previously missing from correctness. Do not reduce these
cases or alter expected outputs to improve a score.

Compilation uses the actual original HIP compiler/extension build, including
both native verification and benchmark binaries where applicable. The original
warmup (10), sample count (100), CUDA-graph/event fallback policy, state reset,
and allocation/timing boundaries remain in the protected harness. Runtime
requirements are the selected ROCm image, a compatible GPU and HIP compiler,
and PyTorch for extension tasks. Generated performance helpers are supplied by
Arena and must not be edited.

Call `python3 scripts/evaluate.py` followed by `validate-task`, or by
`baseline|candidate` and `compile|correctness|performance`. Each action emits one
`ARENA_EVAL_RESULT=` envelope. Missing cases, compiler errors, numerical errors,
and unavailable runtime dependencies are failures, never implicit skips.

Native extension compilation snapshots `src/` into a fresh directory under
`build/native_sources/`. PyTorch hipify writes only into that build copy;
the authored binding, candidate source, and frozen baseline files remain intact.
Relative includes and current candidate bytes are preserved. Staged inputs are
retained with the workspace for inspection. This does not change timed work.

The baseline's declared timing method remains fixed for both roles. If edited
native source fails the current-stream/capture-safety check required by that
method, evaluation rejects it; it cannot force graph timing for an unsafe launch
or downgrade only the candidate to event timing. Implementation/launcher edits
remain within the declared file boundary, and must honor this stream contract.

Performance now compares the actual timed graph outputs to the full protected
CPU reference, poisons those buffers and checks the same graph after replay.
Caller inputs must remain unchanged. Reference calculation and checks remain
outside measured samples; case IDs, input generation, exact-index or numerical
gates, 10 warmups, 100 repetitions and original reset callbacks are retained.
Native void-return entrypoints expose their written output buffers to the
observer; this does not time a separate validation invocation. Integer outputs
use integer-safe poisoning. These are same-input replay checks.
