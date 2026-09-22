# triton_fla_layernorm

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton kernel `layer_norm_fwd_kernel` for maximum GPU throughput
while maintaining numerical correctness.

LayerNorm / RMSNorm with optional SiLU gating (z branch).

Constraints:
- Must maintain the same function signature for `layer_norm_fwd`
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



Protected evaluation checks all three outputs `(out, mean, rstd)`, including
shape, dtype, device and finite values. RMSNorm has no mean output; statistics
are FP32. The original five scored FP32 cases, seeds, affine inputs,
atol=rtol=1e-3, full-wrapper timing, 10 warmups and 100 samples are unchanged.

Additional unscored 2x17 diagnostics use nonuniform, nontrivial weights and bias
and check both LayerNorm/RMSNorm and both SiLU gate orders. The independent
reference computes statistics after gating when `norm_before_gate=False`.
Actual captured outputs are checked; a poisoned replay with changed input,
weight, bias and optional gate must also satisfy the original numerical gate.
Read-only inputs are checked against pristine copies and restored after timing,
even if validation or replay raises. All added checks run outside timing.
