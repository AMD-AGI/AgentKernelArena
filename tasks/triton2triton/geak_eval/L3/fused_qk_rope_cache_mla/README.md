# fused_qk_rope_cache_mla

Optimize the fused QK RoPE + KV cache Triton kernel for AMD MI300X GPU. The kernel fuses query/key RoPE application with KV cache write for MLA attention.

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

`workloads.json` independently records 128 required cases, including all
128 original full-benchmark cases. The original correctness selection
(128 cases) runs first with its original ordering/seeding. Any additional
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


## Complete outputs and measured replay

The original correctness entrypoint returns `True` only after checking all four
outputs and the updated KV cache. The runner requires that boolean result for
each of the original 128 cases. The reference is evaluated from pristine inputs
before the candidate runs; query/key inputs, scales, positions, frequencies,
routing and the caller's original cache are read-only. All returned tensors must
retain their shapes, dtypes and devices. The original `atol=rtol=0.1` gate remains,
including the original FP8-cache conversion to BF16 before comparison and the
full-cache check covering untouched slots.

Timing still measures the original allocating public wrapper with 50 warmups,
200 samples and its original byte-wise cache-slot reset. After timing, the actual
captured four outputs and cache are compared with the same protected reference.
An unscored replay changes query/key inputs, scale, positions and routing, poisons
the outputs and written cache slots, and replays the exact measured invocation.
The original reset executes before the diagnostic cache poison. All caller inputs
and private cache state are restored even if verification fails. The additional
checks do not change the scored input distributions, allocation boundaries or
case list. GPU qualification requires a new complete validator report.
