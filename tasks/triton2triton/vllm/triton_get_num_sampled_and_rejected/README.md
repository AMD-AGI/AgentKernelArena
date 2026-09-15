# triton_get_num_sampled_and_rejected

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton kernel `_get_num_sampled_and_rejected_kernel` for maximum
GPU throughput while maintaining correctness.

The kernel computes per-request num_sampled and num_rejected counts for speculative
decoding. For chunked-prefilling requests (seq_len < prefill_len), both counts are
set to 0. For decode requests, num_rejected = num_logits - num_sampled.

Key optimization opportunities:
- This is a simple element-wise kernel; focus on minimizing memory transactions
- Coalesced memory access patterns

Constraints:
- Must maintain the same function signature for `get_num_sampled_and_rejected`
- Output must match reference exactly (integer counts)


## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
(with one role and one action). Use `ARENA_EVAL_PHASE=candidate_evaluation` for submitted candidates.
`workloads.json` declares all five original cases; the adapter verifies it against the
protected harness table. Original input seeds, comparisons, tolerances, warmups, sample
counts and graph/event timing remain in `scripts/task_runner.py`. Compilation includes
syntax and import/interface checks. Missing candidates, incomplete measurements and
invalid timing fail; commands emit `arena-eval-v1`, never final Arena score reports.
Canonical benchmark helpers must be materialized by Arena; do not edit their generated regions.

