# triton_decode_attn_stage2

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton decode attention stage2 kernel `_fwd_kernel_stage2` for
maximum GPU throughput while maintaining numerical correctness.

The kernel implements the second stage of split-KV decode attention. It
reduces partial attention outputs from stage1 across KV splits:
- Each program handles one (batch, head)
- Iterates over all KV splits, loading partial output and logsumexp
- Combines using the logsumexp trick:
    max_lse = max(lse across splits)
    o = sum(exp(lse_i - max_lse) * mid_o_i) / sum(exp(lse_i - max_lse))
- Outputs final attention output and final logsumexp

Key optimization opportunities:
- Loop unrolling over NUM_KV_SPLITS
- Memory access pattern optimization
- Warp scheduling and occupancy tuning (num_warps, num_stages)
- Vectorized loads/stores

Constraints:
- Must maintain the same function signature for `decode_softmax_reducev_fwd`
- Output must match reference within atol=1e-2, rtol=1e-2 for float16
- Must handle arbitrary head dimensions and number of KV splits


## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
(with one role and one action). Use `ARENA_EVAL_PHASE=candidate_evaluation` for submitted candidates.
`workloads.json` declares all five original cases; the adapter verifies it against the
protected harness table. Original input seeds, comparisons, tolerances, warmups, sample
counts and graph/event timing remain in `scripts/task_runner.py`. Compilation includes
syntax and import/interface checks. Missing candidates, incomplete measurements and
invalid timing fail; commands emit `arena-eval-v1`, never final Arena score reports.
Canonical benchmark helpers must be materialized by Arena; do not edit their generated regions.

