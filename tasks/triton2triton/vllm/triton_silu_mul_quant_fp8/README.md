# triton_silu_mul_quant_fp8

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton SiLU+Mul+FP8 quantization kernel
`_silu_mul_per_token_group_quant_fp8_colmajor` for maximum GPU throughput.

The kernel fuses SiLU activation, element-wise multiplication, and per-group
FP8 quantization with column-major scale output.

Constraints:
- Must maintain the same function signature for `silu_mul_per_token_group_quant_fp8_colmajor`
- Output must match reference within appropriate FP8 tolerance


## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
(with one role and one action). Use `ARENA_EVAL_PHASE=candidate_evaluation` for submitted candidates.
`workloads.json` declares all five original cases; the adapter verifies it against the
protected harness table. Original input seeds, comparisons, tolerances, warmups, sample
counts and graph/event timing remain in `scripts/task_runner.py`. Compilation includes
syntax and import/interface checks. Missing candidates, incomplete measurements and
invalid timing fail; commands emit `arena-eval-v1`, never final Arena score reports.
Canonical benchmark helpers must be materialized by Arena; do not edit their generated regions.


Protected checks validate the complete platform-FP8 output and FP32 column-major
scale pair, shape/device, finite data and positive scales. The existing FP32
SiLU/multiply reference and tolerances are retained: scale atol=1e-2, rtol=1e-1;
dequantized result atol=0.5, rtol=1e-1. The public output-buffer, epsilon and
UE8M0 options have unscored diagnostics; a supplied output must be written and
returned, and UE8M0 scales must be powers of two. Diagnostics retain the public
M%128 and N%256 constraints. All five scored workloads use the original default
mode, source/harness, seeds, 10 warmups, 100 samples and full-wrapper timing.
The exact timed invocation is numerically replayed with changed inputs and both
outputs poisoned. Read-only input is checked and restored even on replay failure.

Additional unscored public-branch controls from PR105: BF16 UE8M0 and FP32 finite/subnormal edges with provided output.
These use explicit `control-upstream-*` manifest rows. Original scored inputs,
numerical gates, seeds, warmups and sample counts remain unchanged.
