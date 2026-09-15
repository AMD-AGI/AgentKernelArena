# triton_gather_block_tables

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton kernel `_gather_block_tables_kernel` for maximum GPU throughput
while maintaining correctness.

The kernel gathers block table rows from a source table to a destination table based
on an index mapping. For each batch index, it copies num_blocks[req_idx] entries from
src_block_table[req_idx] to dst_block_table[batch_idx]. This is simplified from the
original vLLM version to use contiguous block tables instead of pointer-of-pointers.

Key optimization opportunities:
- Block size tuning for memory bandwidth
- Vectorized loads/stores
- Memory coalescing

Constraints:
- Must maintain the same function signature for `gather_block_tables`
- Output must match reference exactly (integer block IDs)


## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
(with one role and one action). Use `ARENA_EVAL_PHASE=candidate_evaluation` for submitted candidates.
`workloads.json` declares all five original cases; the adapter verifies it against the
protected harness table. Original input seeds, comparisons, tolerances, warmups, sample
counts and graph/event timing remain in `scripts/task_runner.py`. Compilation includes
syntax and import/interface checks. Missing candidates, incomplete measurements and
invalid timing fail; commands emit `arena-eval-v1`, never final Arena score reports.
Canonical benchmark helpers must be materialized by Arena; do not edit their generated regions.

