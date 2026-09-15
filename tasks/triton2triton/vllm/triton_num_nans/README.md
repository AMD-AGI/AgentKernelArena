# triton_num_nans

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton kernel `_num_nans_kernel` for maximum GPU throughput
while maintaining correctness.

The kernel counts the number of NaN values in each row of a 2D logits tensor.
It iterates over the vocabulary dimension in blocks, converts to float32,
checks for NaN using libdevice.isnan, and accumulates the count.

Key optimization opportunities:
- Block size tuning for the vocabulary scan
- Efficient reduction of NaN counts
- Minimizing float32 conversions
- Vectorized loads

Constraints:
- Must maintain the same function signature for `get_num_nans`
- Output must match reference exactly (integer NaN counts)


## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
(with one role and one action). Use `ARENA_EVAL_PHASE=candidate_evaluation` for submitted candidates.
`workloads.json` declares all five original cases; the adapter verifies it against the
protected harness table. Original input seeds, comparisons, tolerances, warmups, sample
counts and graph/event timing remain in `scripts/task_runner.py`. Compilation includes
syntax and import/interface checks. Missing candidates, incomplete measurements and
invalid timing fail; commands emit `arena-eval-v1`, never final Arena score reports.
Canonical benchmark helpers must be materialized by Arena; do not edit their generated regions.

