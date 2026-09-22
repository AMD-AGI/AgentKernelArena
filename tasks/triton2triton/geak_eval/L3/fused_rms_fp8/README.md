# fused_rms_fp8

Optimize the fused RMSNorm + FP8 quantization Triton kernel for the configured AMD GPU. The kernel fuses RMSNorm normalization with FP8 quantization.

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

`workloads.json` independently records 32 required cases, including all
25 original full-benchmark cases. The original correctness selection
(25 cases) runs first with its original ordering/seeding. Any additional
benchmark cases then receive the same numerical/output checks. Distinct original
correctness-only cases remain in the manifest; repeated performance configurations
keep their separate indices. No skipped/missing case or incomplete action passes.

Compilation retains the original syntax check. Numerical checks actually execute
the candidate and compare with the protected references; original tolerances are retained with the additional complete-output checks below. Full benchmark input order, seeds, allocation/reset behavior,
warmups, iterations and the canonical helper's median calculation remain unchanged.
The adapter collects fresh device measurements directly from the benchmark calls;
old `build/performance_report.json` files and log text cannot supply evidence.
The original optional reference timings remain diagnostic; Arena uses the frozen
initial implementation's measured times for scoring. No GPU qualification is
implied by the CPU migration checks.

Do not edit `test_kernel_harness.py`, `_contract_oracles.py`, `_timed_contract.py`, `_arena_*.py`,
`workloads.json`, or generated
`_aka_benchmark.py`. The runtime must materialize the canonical benchmark helper
next to the original harness even though the public runner is `_arena_eval.py`.
Unsupported hardware or missing dependencies return a failing envelope; use a
compatible image/GPU before scheduling this task.

## Complete outputs and additional public entrypoints

The original 25 scored shapes, seeds, BF16 inputs, numerical tolerances
(`atol=rtol=0.1`), warmups, iterations and reference timing are unchanged.
The original rounded-residual PyTorch comparison still checks the reconstructed
quantized values, both unquantized normalized outputs and residual output.
Every returned tensor must also have the specified dtype/shape/device and be finite.
Independent FP32 accumulation oracles additionally check raw FP8 values and
positive per-group scales; scales use 10% relative error without a large absolute
allowance. Raw values may differ by at most one representable FP8 step, accounting
for FP32 arithmetic choosing opposite sides of a quantization midpoint. The
original reconstructed-value `atol=rtol=0.1` still applies independently. Compensating an incorrect scale with an incorrect quantized value is
not an acceptable representation of the specified quantization.

Seven additional **unscored** cases exercise all four editable entrypoints:
plain RMS with optional outputs absent; flatten; SiLU-multiply with and without
split-K; reduce-RMS with no split-K and with three/four splits. Controls use signed,
nonuniform inputs and weights, group size 128, and power-of-two channel widths.
Split-K controls also check auxiliary reductions and every returned output.
These controls do not introduce additional timed work or modify score weighting.
The extra wrapper domain uses SiLU, ordinary (not transposed) group scales, and
supplies the third reduction input on split-K reduce-RMS calls. Other library
variants in kernel.py are outside the declared controls and score domain.

All input tensors are read-only. Protected references use pristine copies before
candidate execution. The original actual TimedRun outputs and a perturbed-input
replay of the same captured invocation must both pass the full output checks.
Input mutation fails, and original inputs are restored in finally even on error.
Reference computation, input copies, output poisoning and checks occur outside
measurement. The frozen initial candidate and final candidate follow the same path.
