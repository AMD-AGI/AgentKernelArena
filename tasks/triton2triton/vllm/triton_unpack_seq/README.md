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


Protected checks compare pristine packed data and lengths, enforce output
shape/dtype/device and finite values, and retain the original atol=rtol=1e-3.
Unscored checks cover higher-rank features, zero-length sequences, the second time
block and partial time/feature tiles. All five scored cases and seeds are unchanged.
Performance intentionally retains the original direct JIT launch with reusable
output; it excludes the public wrapper's host lengths.sum().item() and allocation.
Both roles use this same timing unit, grid, launch options, 10 warmups and 100
samples. The actual timed output is numerically checked, poisoned and replayed
with changed packed data and reversed sequence lengths (same total output size).
Inputs and the reusable output buffer are restored even on replay failure.
