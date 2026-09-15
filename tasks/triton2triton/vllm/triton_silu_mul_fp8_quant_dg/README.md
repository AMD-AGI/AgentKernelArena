# triton_silu_mul_fp8_quant_dg

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton `_silu_mul_fp8_quant_deep_gemm` kernel for maximum GPU throughput.
This kernel computes fused SiLU activation + elementwise multiply + FP8 quantization
for DeepGEMM MoE. Input [E, T, 2*H] is split into gate and up projections,
then y = silu(gate) * up is quantized to FP8 with per-group scales.

Key optimization opportunities:
- Pipeline stages for token loop
- Warp count tuning
- Memory access patterns for strided scales

Constraints:
- Must maintain the same function signature for `silu_mul_fp8_quant`
- Dequantized output must match reference within atol=0.5, rtol=0.2
- For AMD GPU use torch.float8_e4m3fnuz


## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
(with one role and one action). Use `ARENA_EVAL_PHASE=candidate_evaluation` for submitted candidates.
`workloads.json` declares all five original cases; the adapter verifies it against the
protected harness table. Original input seeds, comparisons, tolerances, warmups, sample
counts and graph/event timing remain in `scripts/task_runner.py`. Compilation includes
syntax and import/interface checks. Missing candidates, incomplete measurements and
invalid timing fail; commands emit `arena-eval-v1`, never final Arena score reports.
Canonical benchmark helpers must be materialized by Arena; do not edit their generated regions.


Protected checks compare every valid token and every group using the original
FP32 SiLU/multiply reference and unchanged dequantized `atol=0.5, rtol=0.2` gate.
They require the FP8/FP32 output pair, expected shapes/device, finite valid
quantized values and positive finite valid scales. They do not require a new
exact quantization or scale-selection algorithm. Activations and token counts
are read-only. Rows at or beyond an expert's valid count are unspecified, as
both outputs are allocated with `torch.empty`; they are not required to be zero
or finite.

Unscored controls use counts 0/5/7, noncontiguous activation strides, group sizes
128 and 64 with hidden sizes divisible by the group, and nonzero values large
enough to reject an all-zero answer under the original tolerance. A zero group
checks the positive epsilon-scale path. Partial quantization groups are not
claimed as supported by these controls.

Performance keeps the original five scored cases, full-token counts, seed 0,
public wrapper allocations/scale layout, 10 warmups and 100 samples. It captures
both actual `TimedRun` outputs, validates all originally valid tokens, then
changes activations/counts, poisons both outputs and validates the same measured
replay. Read-only inputs are restored in `finally`, including rejected replays.
Correctness keeps the original cases, input scales and seeds 42+i.
