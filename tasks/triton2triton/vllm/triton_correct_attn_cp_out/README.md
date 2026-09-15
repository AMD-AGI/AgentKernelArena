# triton_correct_attn_cp_out

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton kernel `_correct_attn_cp_out_kernel` for maximum GPU
throughput while maintaining numerical correctness.

The kernel performs logsumexp-based correction of attention outputs from
context parallelism. Given N partial log-sum-exp values and one rank's
attention output, it computes the global LSE and rescales the output by
exp(local_lse - global_lse).

Key optimization opportunities:
- Vectorized loads/stores for the head dimension
- Reduction optimization for computing max and sum over N
- Memory access coalescing

Constraints:
- Must maintain the same function signature for `correct_attn_cp_out`
- Output must match reference within atol=1e-2, rtol=1e-2 for float16/float32


## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
(with one role and one action). Use `ARENA_EVAL_PHASE=candidate_evaluation` for submitted candidates.
`workloads.json` declares all five original cases; the adapter verifies it against the
protected harness table. Original input seeds, comparisons, tolerances, warmups, sample
counts and graph/event timing remain in `scripts/task_runner.py`. Compilation includes
syntax and import/interface checks. Missing candidates, incomplete measurements and
invalid timing fail; commands emit `arena-eval-v1`, never final Arena score reports.
Canonical benchmark helpers must be materialized by Arena; do not edit their generated regions.

