# triton_fla_layernorm_gated

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton kernel `layer_norm_gated_fwd_kernel` for maximum GPU throughput
while maintaining numerical correctness.

Layer norm (or RMS norm) with SiLU/sigmoid gated activation.

Constraints:
- Must maintain the same function signature for `layer_norm_gated_fwd`
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


The complete public return contract is `(y, mean, rstd)`: `y` has the input
shape, dtype and device; statistics have shape `(T,)`, FP32 dtype and the input
device. RMSNorm returns `None` for `mean`. Protected checks now validate every
returned value, with finite values and the original atol=rtol=1e-3 numerical
gate. The original five cases cover both normalization modes, both gate
activations, and optional weight/bias configurations; none are removed.

Read-only x, gate, weight and bias are snapshotted before candidate invocation.
The original full wrapper, seeds, 10 warmups and 100 samples remain unchanged.
Actual captured outputs and a poisoned replay after perturbing those inputs
must match the corresponding pristine-input reference. Inputs are restored in
a finally block, including replay failures; all added checks are outside timing.
