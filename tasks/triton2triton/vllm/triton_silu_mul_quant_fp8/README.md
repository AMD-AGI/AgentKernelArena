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

