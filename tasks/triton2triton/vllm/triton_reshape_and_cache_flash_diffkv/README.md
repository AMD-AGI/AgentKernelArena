# triton_reshape_and_cache_flash_diffkv

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton kernel `reshape_and_cache_kernel_flash_diffkv` for maximum
GPU throughput while maintaining numerical correctness.

This kernel is similar to reshape_and_cache_kernel_flash but handles the case
where K and V have different head dimensions. K and V are interleaved per-head
in the cache: [K_head_i | V_head_i] for each head.

Key optimization opportunities:
- Tile size tuning for the target GPU
- Memory access coalescing for scattered writes
- Warp scheduling and occupancy tuning

Constraints:
- Must maintain the same function signature for `reshape_and_cache_flash_diffkv`
- Output must match reference (exact copy for auto dtype)


## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
(with one role and one action). Use `ARENA_EVAL_PHASE=candidate_evaluation` for submitted candidates.
`workloads.json` declares all five original cases; the adapter verifies it against the
protected harness table. Original input seeds, comparisons, tolerances, warmups, sample
counts and graph/event timing remain in `scripts/task_runner.py`. Compilation includes
syntax and import/interface checks. Missing candidates, incomplete measurements and
invalid timing fail; commands emit `arena-eval-v1`, never final Arena score reports.
Canonical benchmark helpers must be materialized by Arena; do not edit their generated regions.

