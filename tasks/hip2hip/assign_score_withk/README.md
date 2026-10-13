# assign_score_withk: native HIP task

Optimize only `src/assign_score_withk_cuda.hip`. All task inputs, references,
CPU comparisons, wrappers, bindings, compiler flags, launch boundaries, and
benchmark helpers are protected. The initial implementation is present and
written in HIP. Arena freezes that initial candidate in a separate workspace
for every baseline action; final candidate actions always build and execute the
submitted implementation, with no baseline fallback.

The complete independent manifest in `workload.json` includes **10 cases**
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


Both timed variants expose their actual outputs to TimedRun. Forward is checked
against the full protected gather/weighted-reduction reference; forward+backward
also checks all score, point-feature and center-feature gradients against CPU
autograd of that reference, using the original `atol=rtol=1e-3`. Poisoned exact
graph replay repeats these checks. The existing stable leaf-gradient buffers
and per-invocation zeroing callback remain inside the same measured boundary;
input values and original gradient state are restored after validation.
The unused scalar CPU oracle's extra point-feature term is corrected to match
the documented neighbor difference and independently tested vectorized oracle.
No shape, seed, mode, warmup or sample was removed.

Correctness additionally checks all three backward gradients with nonuniform
upstream gradients against an independent analytic CPU oracle, frozen before
the candidate runs. Original sum-gradient timing and its gates are unchanged.
