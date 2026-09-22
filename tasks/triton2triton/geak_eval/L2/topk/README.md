# topk

Optimize the Top-K selection Triton kernel for AMD MI300X GPU. The kernel implements 1-stage (small M) and 2-stage (large M) selection.

CRITICAL CONSTRAINTS:
- Your optimized kernel MUST handle ALL test configurations in the full benchmark (--full-benchmark), not only those in the quick benchmark. The full benchmark tests approximately 80 configurations including very large shapes.
- If you dispatch small shapes to PyTorch fallbacks, the Triton kernel must still correctly handle all remaining shapes. Do not assume the quick benchmark covers all cases.
- Test with --full-benchmark before finalizing to catch shape-coverage gaps.

## Candidate and baseline contract

The actual initial `kernel.py` functions are implemented. Optimize the declared
function scopes (and permitted implementation helpers); keep other functions,
references, input builders, parameter tables and output interfaces unchanged.
Some declared entrypoints are Python launch functions over Triton kernels; keep
their calling convention. A final empty or missing entrypoint is an error, never
a request to run a reference implementation instead.

Arena creates an independent frozen `initial_candidate` workspace for baseline
measurement. Both roles execute this task's local `kernel.py`; no `GEAK_WORK_DIR`
or external worktree selects a candidate or baseline. Installed PyTorch/Triton
(and AITER utilities where imported by the existing source) are runtime image
dependencies. This task neither clones dependencies nor imports Arena/agent code.

## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or
`python3 _arena_eval.py baseline|candidate compile|correctness|performance`, selecting
one role and action. The runner emits one `ARENA_EVAL_RESULT=` envelope using
`arena-eval-v1`. Arena owns final reports and scoring.

`workloads.json` independently records 82 required cases, including all
80 original full-benchmark cases. The original correctness selection
(80 cases) runs first with its original ordering/seeding. Any additional
benchmark cases then receive the same numerical/output checks. Distinct original
correctness-only cases remain in the manifest; repeated performance configurations
keep their separate indices. No skipped/missing case or incomplete action passes.

Compilation retains the original syntax check. Numerical checks actually execute
the candidate and compare with the protected references; original numerical tolerances are retained; the additional output checks below
are required. Full benchmark input order, seeds, allocation/reset behavior,
warmups, iterations and the canonical helper's median calculation remain unchanged.
The adapter collects fresh device measurements directly from the benchmark calls;
old `build/performance_report.json` files and log text cannot supply evidence.
The original optional reference timings remain diagnostic; Arena uses the frozen
initial implementation's measured times for scoring. No GPU qualification is
implied by the CPU migration checks.

Do not edit `test_kernel_harness.py`, `_arena_*.py`, `_timed_contract.py`, `workloads.json`, or generated
`_aka_benchmark.py`. The runtime must materialize the canonical benchmark helper
next to the original harness even though the public runner is `_arena_eval.py`.
Unsupported hardware or missing dependencies return a failing envelope; use a
compatible image/GPU before scheduling this task.

## Complete output and replay checks

Correctness computes its reference from private input snapshots before invoking
the candidate. All read-only inputs retain their original bytes, shape, strides,
dtype and device. Every output must have its declared shape, dtype and device
and contain only finite values; casts cannot hide an invalid public dtype.
Performance retains every original case, seed, allocation boundary, warmup,
sample count and median/adaptive-repeat policy. It additionally compares the
actual `TimedRun.outputs`, then perturbs inputs in place, poisons output buffers
and validates the same captured invocation through `TimedRun.rerun()`.
The reference, snapshots and replay checks run outside device timing. Inputs
are restored even on failure. `_timed_contract.py` is protected task code.

The original `atol=1e-4 * hidden`, `rtol=1.3e-6` comparisons remain, together
with exact selection checks: returned values equal their indexed input values,
match the descending top-k value multiset, and use distinct in-range indices.
Tied maxima permit any distinct valid indices. Two additional correctness-only
known-answer cases exercise ties/negative values and the noncontiguous wrapper
path across both one-stage and two-stage dispatch. The 80 scored cases are
unchanged; these controls add no timing rows.
