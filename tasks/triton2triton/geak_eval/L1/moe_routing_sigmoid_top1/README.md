# moe_routing_sigmoid_top1

Optimize the fused sigmoid-gated top-1 MOE routing Triton kernel for AMD MI300X GPU. The kernel fuses GEMM (X @ W), sigmoid gating, and top-1 selection.

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

`workloads.json` independently records 34 required cases, including all
34 original full-benchmark cases. The original correctness selection
(34 cases) runs first with its original ordering/seeding. Any additional
benchmark cases then receive the same numerical/output checks. Distinct original
correctness-only cases remain in the manifest; repeated performance configurations
keep their separate indices. No skipped/missing case or incomplete action passes.

Compilation retains the original syntax check. Numerical checks actually execute
the candidate and compare with the protected references; original numerical tolerances are retained, with the full-output checks below. Full benchmark input order, seeds, allocation/reset behavior,
warmups, iterations and the canonical helper's median calculation remain unchanged.
The adapter collects fresh device measurements directly from the benchmark calls;
old `build/performance_report.json` files and log text cannot supply evidence.
The original optional reference timings remain diagnostic; Arena uses the frozen
initial implementation's measured times for scoring. No GPU qualification is
implied by the CPU migration checks.

Do not edit `test_kernel_harness.py`, `_timed_contract.py`, `_arena_*.py`, `workloads.json`, or generated
`_aka_benchmark.py`. The runtime must materialize the canonical benchmark helper
next to the original harness even though the public runner is `_arena_eval.py`.
Unsupported hardware or missing dependencies return a failing envelope; use a
compatible image/GPU before scheduling this task.

## Complete output and measured invocation checks

Both routing IDs and FP32 weights are checked, including shared-expert fields.
The original integer-input oracle and `atol=rtol=1e-4` gates remain intact.
For the original random noninteger timing inputs, the independent oracle uses
FP32 dot accumulation before sigmoid, matching the operator rather than rounding
the dot to BF16 first. Ties select the leftmost expert. Original reference timing
is retained as diagnostic timing. Two unscored controls cover no shared expert
and an exact sigmoid tie. There are 36 correctness and 34 unchanged scored cases.

Correctness uses pristine, independent oracle inputs prepared before the candidate.
Performance retains the original helper, warmups, repetitions, allocation and
scored case order. After timing, the actual `TimedRun.outputs` must match that
oracle; the same captured invocation is then replayed after changing inputs in
place and poisoning writable outputs. Its complete outputs must match a newly
computed private oracle. Every read-only input is checked byte-for-byte and
restored in `finally`, including failure paths. These checks run outside timing
and identically for the frozen initial candidate and submitted candidate.
