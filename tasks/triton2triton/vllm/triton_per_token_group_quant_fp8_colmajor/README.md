# triton_per_token_group_quant_fp8_colmajor

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton per-token-group FP8 quantization kernel with column-major
scale output `_per_token_group_quant_fp8_colmajor` for maximum GPU throughput.

The kernel computes per-group max, scales, and quantizes to FP8 format,
storing scales in column-major layout.

Constraints:
- Must maintain the same function signature for `per_token_group_quant_fp8_colmajor`
- Quantized output must match reference within appropriate FP8 tolerance


## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
(with one role and one action). Use `ARENA_EVAL_PHASE=candidate_evaluation` for submitted candidates.
`workloads.json` declares all five original cases; the adapter verifies it against the
protected harness table. Original input seeds, comparisons, tolerances, warmups, sample
counts and graph/event timing remain in `scripts/task_runner.py`. Compilation includes
syntax and import/interface checks. Missing candidates, incomplete measurements and
invalid timing fail; commands emit `arena-eval-v1`, never final Arena score reports.
Canonical benchmark helpers must be materialized by Arena; do not edit their generated regions.


Protected checks enforce both output shapes, the platform FP8 dtype, FP32 scales,
finite values, positive scales, and column-major scale strides. The original scale gate remains
atol=1e-5, rtol=1e-3; the dequantized output gate remains atol=rtol=0.1.
References read pristine input. Unscored diagnostics cover padded rows, group-size
17 tails, zero groups, a non-default epsilon, and the public power-of-two scale mode.
All five scored cases and seeds remain unchanged and time the original default
quantization mode. The exact timed invocation is observed and replayed with changed
input and poisoned outputs. Input and hooks are restored even on failure; 10 warmups,
100 samples, full-wrapper timing and original source/harness are preserved.
