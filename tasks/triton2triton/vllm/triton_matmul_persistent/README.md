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

