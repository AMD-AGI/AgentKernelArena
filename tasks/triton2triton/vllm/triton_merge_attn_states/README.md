# triton_merge_attn_states

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton kernel `merge_attn_states_kernel` for maximum GPU throughput
while maintaining numerical correctness.

The kernel merges prefix and suffix partial attention outputs using logsumexp
rescaling (section 2.2 of https://arxiv.org/pdf/2501.01005). It computes:
  out = (exp(lse_p - max_lse) * prefix_out + exp(lse_s - max_lse) * suffix_out)
        / (exp(lse_p - max_lse) + exp(lse_s - max_lse))

Key optimization opportunities:
- Vectorized loads/stores for the head dimension
- Memory access coalescing
- Warp scheduling and occupancy tuning

Constraints:
- Must maintain the same function signature for `merge_attn_states`
- Output must match reference within atol=1e-2, rtol=1e-2 for float16


## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
(with one role and one action). Use `ARENA_EVAL_PHASE=candidate_evaluation` for submitted candidates.
`workloads.json` declares all five original cases; the adapter verifies it against the
protected harness table. Original input seeds, comparisons, tolerances, warmups, sample
counts and graph/event timing remain in `scripts/task_runner.py`. Compilation includes
syntax and import/interface checks. Missing candidates, incomplete measurements and
invalid timing fail; commands emit `arena-eval-v1`, never final Arena score reports.
Canonical benchmark helpers must be materialized by Arena; do not edit their generated regions.

