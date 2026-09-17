# triton_per_token_group_quant_int8

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton per-token-group INT8 quantization kernel
`_per_token_group_quant_int8` for maximum GPU throughput.

The kernel computes per-group max, scales to [-127, 127], and quantizes to int8.

Constraints:
- Must maintain the same function signature for `per_token_group_quant_int8`
- Output must match reference within atol=1, rtol=0 for int8 values (the existing harness gate)


## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
(with one role and one action). Use `ARENA_EVAL_PHASE=candidate_evaluation` for submitted candidates.
`workloads.json` declares all five original cases; the adapter verifies it against the
protected harness table. Original input seeds, comparisons, tolerances, warmups, sample
counts and graph/event timing remain in `scripts/task_runner.py`. Compilation includes
syntax and import/interface checks. Missing candidates, incomplete measurements and
invalid timing fail; commands emit `arena-eval-v1`, never final Arena score reports.
Canonical benchmark helpers must be materialized by Arena; do not edit their generated regions.


Protected checks require the complete `(quantized, scales)` pair with INT8 codes,
FP32 positive finite scales, and the original shape/device contracts. The original
one-code allowance is retained; scale comparison remains atol=1e-4, rtol=1e-3.
The reference reads pristine input before any candidate call. An unscored diagnostic
covers higher-rank inputs, zero data and partial launch tiles with a non-default epsilon floor.
Performance observes both outputs from the actual timed invocation, changes input
sign and magnitude, poisons both captured outputs, and numerically checks the same
replay. Inputs and module hooks are restored even on failure. All five scored cases,
seeds, original source/harness, full-wrapper timing, 10 warmups and 100 samples stay
unchanged; diagnostics do not add measurements to the score.

Additional unscored public-branch controls from PR105: Exact FP16 extrema/subnormal/half-step quantization and non-power-of-two epsilon controls.
Their `control-upstream-*` manifest rows preserve all existing scored cases,
numerical gates, seeds and timing.
