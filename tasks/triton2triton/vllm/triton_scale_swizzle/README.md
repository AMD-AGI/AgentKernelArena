# triton_scale_swizzle

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton scale swizzle kernel `triton_scale_swizzle` for maximum
GPU throughput while maintaining numerical correctness.

The kernel rearranges tensor data from row-major to block-scaled swizzle format,
suitable for NVIDIA TMEM block scaling.

Constraints:
- Must maintain the same function signature for `triton_mx_block_rearrange`
- Output must match reference exactly (bit-exact for uint8 data)


## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
(with one role and one action). Use `ARENA_EVAL_PHASE=candidate_evaluation` for submitted candidates.
`workloads.json` declares all five original cases; the adapter verifies it against the
protected harness table. Original input seeds, comparisons, tolerances, warmups, sample
counts and graph/event timing remain in `scripts/task_runner.py`. Compilation includes
syntax and import/interface checks. Missing candidates, incomplete measurements and
invalid timing fail; commands emit `arena-eval-v1`, never final Arena score reports.
Canonical benchmark helpers must be materialized by Arena; do not edit their generated regions.

