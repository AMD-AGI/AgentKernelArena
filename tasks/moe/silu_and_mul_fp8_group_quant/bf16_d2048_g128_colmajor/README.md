# SiLU-and-mul with FP8 group quantization, bfloat16, d = 2048, group 128, column-major scales

The epilogue between the two GEMMs of a block-FP8 MoE expert (the vLLM/DeepGEMM path): `y = silu(gate) * up`
on a `[n_tokens, 2*2048]` bfloat16 input laid out `[gate | up]`, then per-token groups of 128 columns are
quantized to `float8_e4m3fn` with `scale = max(absmax, 1e-10) / 448` and codes clamped to +-448 (the OCP
format used on gfx950). The scales are written column-major, as the FP8 grouped GEMM expects: `out_scales`
is declared `[n_groups, n_tokens]`, which is the same memory as the `[n_tokens, n_groups]` column-major
matrix of the original interface. `d = inter_size / tp`. The token axis is a multiple of 128 because the
upstream launcher asserts it.

The reference is float64 with no intermediate rounding. The production kernel narrows the SiLU and the
product to bfloat16, which is exactly the error the default baseline-relative bound measures on the
baseline; a candidate that rounds differently stays within twice that error. Codes and scales are floats,
so they use the default bound. The code comparison also asks the baseline for at least 25 dB SQNR; the
FP8 format gives about 32 dB.

## Baseline

vLLM `_silu_mul_per_token_group_quant_fp8_colmajor` from
`vllm/model_executor/layers/quantization/utils/fp8_utils.py` at commit
`edbc4969a76b25c531f7b9bb16b79984e65ef023`, kernel unchanged. The launcher is fixed to this scenario (silu:
`alpha=1`, `beta=0`, no clamp, no UE8M0) and writes into the preallocated outputs. Each program handles
8 rows by 128 columns.

The original task's torch2flydsl variant wrote row-major scales; this task keeps the vLLM column-major
layout. The old task's small shapes (128 to 512 rows) run far below 10 us, so performance cases start at
4096 rows (about 42 MB moved) and span 128x; the largest case holds 4.3 GB of input and the float64
reference works in row chunks. Smaller slices and the boundary cases are correctness-only.

## Scope

The original tasks also exercised float16 and float32 inputs, a non-default epsilon, UE8M0 (power-of-two)
scales, a clamp limit and row-major scales. A scenario fixes dtype and layout, so those are separate
scenarios (or are not part of the production path of this epilogue); this task covers the bfloat16 SiLU,
no-clamp, linear-scale, column-major configuration only.
