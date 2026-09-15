# fused_mxfp4_quant_moe_sort

Optimize the fused MXFP4 quantization + MOE sort Triton kernel for AMD MI300X GPU. Fuses dynamic microscaling FP4 quantization with MOE token sorting.

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

`workloads.json` independently records 24 required cases, including all
24 original full-benchmark cases. The original correctness selection
(24 cases) runs first with its original ordering/seeding. Any additional
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

The harness supports both standalone text streams and Arena's captured action
logs. It enables line buffering only when the output stream exposes
`reconfigure`; log capture does not require that terminal-specific method.

## Reference correction found during migration

An independent E8M0 known-answer check found incorrect NaN decoding: code `134`
was treated as NaN, and code `255` was treated as infinity. The quant/sort helper
also subtracted the bias in uint8, so code `126` wrapped instead of producing
`0.5`. Decode the exponent after conversion to float32 and reserve code `255` for
NaN, following [OCP MX specification section 5.4.1/table 7](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf).
CPU regression checks cover all 256 encodings against PyTorch's native E8M0
conversion. Original numerical tolerances and performance code are unchanged;
GPU qualification remains required.
