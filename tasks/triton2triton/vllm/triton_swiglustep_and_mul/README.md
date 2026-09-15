# triton_swiglustep_and_mul

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton SwiGLU-step-and-mul activation kernel `_swiglustep_and_mul_kernel`
for maximum GPU throughput while maintaining numerical correctness.

The kernel computes: silu(x[:,:d]).clamp(max=limit) * x[:,d:].clamp(-limit, limit)
where d = input_dim // 2.

Key optimization opportunities:
- Block size tuning
- Memory access pattern optimization
- Fused computation optimizations

Constraints:
- Must maintain the same function signature for `swiglustep_and_mul`
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

