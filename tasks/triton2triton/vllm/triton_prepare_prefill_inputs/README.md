# triton_prepare_prefill_inputs

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton kernel `_prepare_prefill_inputs_kernel` for maximum GPU throughput
while maintaining correctness.

The kernel copies token IDs from a 2D all_token_ids buffer into a flat input_ids tensor
for each prefilling request. It uses idx_mapping to map batch indices to request state
indices and query_start_loc for per-request output offsets. It also stores the next
prefill token if the prefill is not yet complete.

Key optimization opportunities:
- Block size tuning for memory coalescing
- Vectorized loads/stores
- Memory access pattern optimization

Constraints:
- Must maintain the same function signature for `prepare_prefill_inputs`
- Output must match reference exactly (integer token IDs)


## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
(with one role and one action). Use `ARENA_EVAL_PHASE=candidate_evaluation` for submitted candidates.
`workloads.json` declares all five original cases; the adapter verifies it against the
protected harness table. Original input seeds, comparisons, tolerances, warmups, sample
counts and graph/event timing remain in `scripts/task_runner.py`. Compilation includes
syntax and import/interface checks. Missing candidates, incomplete measurements and
invalid timing fail; commands emit `arena-eval-v1`, never final Arena score reports.
Canonical benchmark helpers must be materialized by Arena; do not edit their generated regions.

