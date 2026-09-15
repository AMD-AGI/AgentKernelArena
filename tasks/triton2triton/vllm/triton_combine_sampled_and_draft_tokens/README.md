# triton_combine_sampled_and_draft_tokens

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton kernel `_combine_sampled_and_draft_tokens_kernel` for maximum
GPU throughput while maintaining correctness.

The kernel combines last sampled tokens and draft tokens into the input_ids buffer
for speculative decoding. It also computes logits_indices mapping. For each decode
request, it writes the last sampled token followed by draft tokens into the
appropriate positions in input_ids.

Key optimization opportunities:
- Block size tuning
- Vectorized loads/stores for draft tokens
- Reducing conditional branches

Constraints:
- Must maintain the same function signature for `combine_sampled_and_draft_tokens`
- Output must match reference exactly (integer token IDs and indices)


## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
(with one role and one action). Use `ARENA_EVAL_PHASE=candidate_evaluation` for submitted candidates.
`workloads.json` declares all five original cases; the adapter verifies it against the
protected harness table. Original input seeds, comparisons, tolerances, warmups, sample
counts and graph/event timing remain in `scripts/task_runner.py`. Compilation includes
syntax and import/interface checks. Missing candidates, incomplete measurements and
invalid timing fail; commands emit `arena-eval-v1`, never final Arena score reports.
Canonical benchmark helpers must be materialized by Arena; do not edit their generated regions.

