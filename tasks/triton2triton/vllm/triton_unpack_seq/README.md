# triton_unpack_seq

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton kernel `_unpack_seq_triton_kernel` for maximum GPU
throughput while maintaining numerical correctness.

The kernel unpacks a batched [B, Lmax, D] tensor back to contiguous
variable-length sequences in a flat [N, D] tensor, effectively removing
padding.

Key optimization opportunities:
- Block size tuning (BLOCK_T, BLOCK_D) for the target GPU
- Memory access coalescing
- Warp scheduling and occupancy tuning

Constraints:
- Must maintain the same function signature for `unpack_seq`
- Output must match reference exactly


## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
(with one role and one action). Use `ARENA_EVAL_PHASE=candidate_evaluation` for submitted candidates.
`workloads.json` declares all five original cases; the adapter verifies it against the
protected harness table. Original input seeds, comparisons, tolerances, warmups, sample
counts and graph/event timing remain in `scripts/task_runner.py`. Compilation includes
syntax and import/interface checks. Missing candidates, incomplete measurements and
invalid timing fail; commands emit `arena-eval-v1`, never final Arena score reports.
Canonical benchmark helpers must be materialized by Arena; do not edit their generated regions.

