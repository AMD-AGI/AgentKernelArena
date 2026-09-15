# triton_fused_gdn_gating

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton `fused_gdn_gating_kernel` for maximum GPU throughput.
This kernel computes gating values for Gated Delta Net attention:
  g = -exp(A_log) * softplus(a + dt_bias)
  beta_output = sigmoid(b)

Key optimization opportunities:
- Block size tuning for the heads dimension
- Vectorized loads/stores
- num_warps tuning
- Minimizing type conversions

Constraints:
- Must maintain the same function signature for `fused_gdn_gating`
- Output must match reference within atol=1e-3, rtol=1e-3


## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
(with one role and one action). Use `ARENA_EVAL_PHASE=candidate_evaluation` for submitted candidates.
`workloads.json` declares all five original cases; the adapter verifies it against the
protected harness table. Original input seeds, comparisons, tolerances, warmups, sample
counts and graph/event timing remain in `scripts/task_runner.py`. Compilation includes
syntax and import/interface checks. Missing candidates, incomplete measurements and
invalid timing fail; commands emit `arena-eval-v1`, never final Arena score reports.
Canonical benchmark helpers must be materialized by Arena; do not edit their generated regions.



Protected checks require the exact `(g, beta_output)` interface: `[1, batch,
num_heads]` tensors, FP32 `g`, `beta_output` with the dtype of `b`, correct
devices and finite values. Both results use the existing independent CPU
softplus/exp/sigmoid reference and atol=rtol=1e-3 gate. The five scored cases,
seed 42, FP16 a/b inputs, full wrapper, 10 warmups and 100 samples are unchanged.

An unscored 2x11-head diagnostic checks the masked tail, softplus beta=.5 and
threshold=2, and finite inputs on both sides of the stable softplus branch.
Actual timed outputs and a poisoned replay after changing all four inputs must
match the reference. Inputs are compared to pristine copies and restored even
on replay or validation failure. Added checks are outside the timed interval.
