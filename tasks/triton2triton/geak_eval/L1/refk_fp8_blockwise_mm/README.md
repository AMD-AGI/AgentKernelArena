# refk_fp8_blockwise_mm

Optimize the FP8 block-scale GEMM kernel for AMD MI325X GPU. The kernel performs block-wise dequantization of FP8 inputs with per-block scaling factors, followed by matrix multiplication. Focus on fusing dequantization with the GEMM and optimizing memory access patterns for the block-scale layout.

CRITICAL CONSTRAINTS:
- DO NOT use @triton.autotune. This kernel is benchmarked across 29 shapes, and autotune causes compilation explosion (N_configs * 29 compilations) that exceeds the evaluation timeout.
- Instead, use heuristic config selection: pick BLOCK_M/BLOCK_N/BLOCK_K based on matrix dimensions at launch time (e.g., via if/elif on M, N, K).
- For small shapes where Triton overhead dominates, consider dispatching to torch.mm with BF16 casting as a fast path.
- For large shapes, tile to fit in L2 cache (4 MB per CU on MI300X).

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

`workloads.json` independently records 29 required cases, including all
29 original full-benchmark cases. The original correctness selection
(25 cases) runs first with its original ordering/seeding. Any additional
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
