# triton_matmul_persistent

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton persistent matmul kernel `matmul_kernel_persistent` for maximum
GPU throughput while maintaining numerical correctness.

The kernel implements a persistent tiling strategy for matrix multiplication (C = A @ B + bias)
with support for large tensors (>2^31 elements), optional bias, and grouped tile scheduling.

Key optimization opportunities:
- Tile size tuning (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)
- Memory access pattern optimization
- Warp scheduling and occupancy tuning (num_warps, num_stages)
- Persistent kernel loop optimization

Constraints:
- Must maintain the same function signature for `matmul_persistent`
- Output must match reference within atol=1e-2, rtol=1e-2 for float16


## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
(with one role and one action). Use `ARENA_EVAL_PHASE=candidate_evaluation` for submitted candidates.
`workloads.json` declares all five original cases; the adapter verifies it against the
protected harness table. Original input seeds, comparisons, tolerances, warmups, sample
counts and graph/event timing remain in `scripts/task_runner.py`. Compilation includes
syntax and import/interface checks. Missing candidates, incomplete measurements and
invalid timing fail; commands emit `arena-eval-v1`, never final Arena score reports.
Canonical benchmark helpers must be materialized by Arena; do not edit their generated regions.


The protected checks validate the full FP16 result against pristine FP32 matmul
(with optional bias added before the FP16 cast), at the original `atol=rtol=1e-2`.
They require the expected shape, dtype, device and finite values, and preserve all
read-only operands. Unscored diagnostics cover M129/N259/K67 with strided A/B and
bias, and M1153/N8449/K16 with 340 original FP16 tiles for grouped scheduling and
multiple persistent waves on gfx950. These do not replace any scored case or
certify the greater-than-2^31 index path or unscored dtypes.

Performance checks use the actual output from the original measured public call.
After timing, they compare it against pristine inputs, change both A and B,
poison the captured output and numerically verify the exact `TimedRun` replay.
Inputs are restored in `finally`, including rejected or exceptional replays.
Original seeds (correctness 42+i, performance 0), five cases, allocations,
10 warmups and 100 samples remain unchanged for baseline and candidate.
