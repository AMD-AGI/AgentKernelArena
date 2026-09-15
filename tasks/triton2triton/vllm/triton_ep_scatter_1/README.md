# triton_ep_scatter_1

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton `_fwd_kernel_ep_scatter_1` kernel for maximum GPU throughput.
This kernel computes expert start locations via prefix sum of 128-aligned token
counts, then fills m_indices with expert IDs.

Key optimization opportunities:
- Efficient prefix sum computation
- m_indices fill loop optimization

Constraints:
- Must maintain the same function signature for `ep_scatter_1`
- expert_start_loc and m_indices must exactly match reference


## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
(with one role and one action). Use `ARENA_EVAL_PHASE=candidate_evaluation` for submitted candidates.
`workloads.json` declares all five original cases; the adapter verifies it against the
protected harness table. Original input seeds, comparisons, tolerances, warmups, sample
counts and graph/event timing remain in `scripts/task_runner.py`. Compilation includes
syntax and import/interface checks. Missing candidates, incomplete measurements and
invalid timing fail; commands emit `arena-eval-v1`, never final Arena score reports.
Canonical benchmark helpers must be materialized by Arena; do not edit their generated regions.

