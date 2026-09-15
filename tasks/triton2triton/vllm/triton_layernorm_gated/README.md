# triton_layernorm_gated

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton kernel `_layer_norm_fwd_1pass_kernel` for maximum GPU throughput.
Supports LayerNorm and RMSNorm with optional SiLU gating (z * sigmoid(z)).
Gate can be applied before or after normalization (NORM_BEFORE_GATE flag).
Constraints:
- Must maintain the same function signature for `layer_norm_fwd`
- Output must match reference within atol=1e-2, rtol=1e-2


## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
(with one role and one action). Use `ARENA_EVAL_PHASE=candidate_evaluation` for submitted candidates.
`workloads.json` declares all five original cases; the adapter verifies it against the
protected harness table. Original input seeds, comparisons, tolerances, warmups, sample
counts and graph/event timing remain in `scripts/task_runner.py`. Compilation includes
syntax and import/interface checks. Missing candidates, incomplete measurements and
invalid timing fail; commands emit `arena-eval-v1`, never final Arena score reports.
Canonical benchmark helpers must be materialized by Arena; do not edit their generated regions.



Protected checks validate the full `(out, mean, rstd)` tuple, including shape,
dtype, device and finite values. Statistics are FP32 in group-major row order;
RMSNorm returns no mean. All five original scored FP16 cases, seeds, random
weights/bias, atol=rtol=1e-2, full wrapper, 10 warmups and 100 samples remain.

Additional unscored 2x34 checks exercise both advertised gate orders, both
normalizations, the existing group_size=17 interface and a caller-provided out
buffer. An independent CPU reference normalizes each group separately and checks
that the supplied output buffer is used. Actual captured outputs and a poisoned
replay after input/affine/gate perturbation must also pass. Read-only inputs are
verified against pristine copies and restored even when replay fails. These
checks add no scored cases and run outside timing.
