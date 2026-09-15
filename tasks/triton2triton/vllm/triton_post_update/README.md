# triton_post_update

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton kernel `_post_update_kernel` for maximum GPU throughput
while maintaining correctness.

The kernel performs post-sampling updates for speculative decoding: updates
last_sampled_tokens, total_len, output_bin_counts (histogram), all_token_ids,
and num_computed_tokens for each request. Uses num_warps=1 for sequential
per-request execution due to data dependencies within each request.

IMPORTANT: This kernel must use num_warps=1 because it has sequential
dependencies within each program (loop over sampled tokens with read-modify-write
on output_bin_counts).

Key optimization opportunities:
- Memory access pattern optimization
- Reducing redundant loads
- Register reuse

Constraints:
- Must maintain the same function signature for `post_update`
- Must preserve num_warps=1
- Output must match reference exactly (integer values)


## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
(with one role and one action). Use `ARENA_EVAL_PHASE=candidate_evaluation` for submitted candidates.
`workloads.json` declares all five original cases; the adapter verifies it against the
protected harness table. Original input seeds, comparisons, tolerances, warmups, sample
counts and graph/event timing remain in `scripts/task_runner.py`. Compilation includes
syntax and import/interface checks. Missing candidates, incomplete measurements and
invalid timing fail; commands emit `arena-eval-v1`, never final Arena score reports.
Canonical benchmark helpers must be materialized by Arena; do not edit their generated regions.

