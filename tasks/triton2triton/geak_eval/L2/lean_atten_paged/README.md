# lean_atten_paged

Optimize the Lean Attention + Paged Attention Triton decode kernel for AMD MI300X GPU. The kernel uses persistent streaming-k with tile-level scheduling.

KEY OPTIMIZATION OPPORTUNITY:
- The lock buffer used for cross-tile synchronization is currently allocated per kernel launch. Pre-allocating the lock buffer outside the kernel (in the host wrapper) eliminates per-launch allocation overhead.
- The lock buffer synchronizes across thread blocks during the streaming-k reduction. Moving its allocation to one-time initialization reduces a significant latency source in the launch-critical path.
- Also consider optimizing the tile scheduling and reduction patterns within the persistent kernel body.

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

`workloads.json` independently records 11 required cases, including all
7 original full-benchmark cases. The original correctness selection
(4 cases) runs first with its original ordering/seeding. Any additional
benchmark cases then receive the same numerical/output checks. Distinct original
correctness-only cases remain in the manifest; repeated performance configurations
keep their separate indices. No skipped/missing case or incomplete action passes.

Compilation retains the original syntax check. Numerical checks actually execute
the candidate and compare with the protected references; tolerances and output
checks are unchanged. Full benchmark input order, seeds, allocation/reset behavior,
warmups, iterations and the canonical helper's median calculation remain unchanged.
The adapter collects fresh device measurements directly from the benchmark calls;
old `build/performance_report.json` files and log text cannot supply evidence.
The original optional reference timings remain diagnostic; Arena uses the frozen
initial implementation's measured times for scoring. No GPU qualification is
implied by the CPU migration checks.

Do not edit `test_kernel_harness.py`, `_arena_*.py`, `workloads.json`, or generated
`_aka_benchmark.py`. The runtime must materialize the canonical benchmark helper
next to the original harness even though the public runner is `_arena_eval.py`.
Unsupported hardware or missing dependencies return a failing envelope; use a
compatible image/GPU before scheduling this task.

The protected adapter now checks the output of the actual measured invocation,
then perturbs Q/K/V, poisons output and intermediate scratch, and verifies a
replay against the same paged reference. Read-only page mappings and inputs are
checked; input and scratch state are restored even after failure. The original
lock-zeroing preparation remains outside timing and is reused for replay.
The original 11 correctness cases, 7 performance cases, 50 warmups, 200 samples,
median reduction, numerical thresholds and timed wrapper remain unchanged.
