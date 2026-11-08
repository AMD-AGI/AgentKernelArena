# Fused bf16 router-gate GEMM + MXFP8 activation quantization, DeepSeek-V3/R1 (N = 256, K = 7168)

One launch computes `y = x @ w^T` (bfloat16 `x` [m, 7168], router weight `w` [256, 7168], float32 accumulation, bfloat16 output, no bias) and the
dynamic MXFP8 quantization of `x`: `x_quant` [m, 7168] float8_e4m3fn codes and `x_scale` [m, 224] uint8 E8M0 scales, one per block of 32 values
along the hidden axis. This is AITER's `fused_gemm_a16w16_quant_x`; its docstring says it fuses the GEMM "with the activation per-1x32 MXFP8
quantization that immediately follows the router-gate GEMM in MoE flows", saving a separate pass that casts `x` from bfloat16 to fp8 and e8m0 scales.

**Scenario and config citations.** DeepSeek-V3 `config.json` (Hugging Face; DeepSeek-R1 has the same architecture fields): `hidden_size` 7168 (K) and
`n_routed_experts` 256 (N, the router gate `mlp.gate.weight` [256, 7168]). AITER's test for this operator
(`op_tests/triton_tests/gemm/fused/test_fused_gemm_a16w16_quant_x.py`) names the shape "DSR1 router GEMM: num_tokens x 256 x 7168" and ships a tuned file for it
(`configs/gemm/gfx950-GEMM-A16W16-N=256-K=7168.json`, inlined in the baseline). The same test also lists a 384-expert router (N = 384, K = 7168) that it calls
"DSv4"; that model is not cited here because its configuration could not be checked against a public file, so only the DeepSeek-R1 router is a task scenario. The
MXFP8 activation format (1x32 blocks, E8M0 scales) feeds an MXFP8 expert GEMM downstream; no pinned vLLM or SGLang call site of the fused operator exists.

**Reference.** `reference.py` is float64: `y = x @ w^T`, and the quantization of `x` with the exact rule of the operator: amax of the 32-block is rounded to a
power of two by adding a quarter of the mantissa field (mantissa >= 1.75 rounds up), the unbiased scale exponent is `floor(log2(amax rounded)) - 8` clamped to
[-127, 127] (so the rounded amax maps to 2^8 and nothing saturates the E4M3 maximum of 448), the elements are multiplied by the exact power of two 2^-exponent
and the evaluator rounds the float32 result to float8_e4m3fn (round to nearest even). It is the same rule as the sibling task
`tasks/triton/quantization/dynamic_mxfp8_quant`, whose reference it mirrors; it is independent of the baseline.

**Comparison.** `y` uses the default `baseline_relative` bound. `x_quant` and `x_scale` are `exact`: the scaled value is exactly representable in float32 and a
single round-to-nearest-even conversion is deterministic (the AITER test allows one code of difference for the fp8 cast; this task does not, as the sibling
quantization task does not). All-zero blocks are not covered (the production op yields an undefined scale there) and the normal inputs never produce them.

**Inputs.** `x` is standard normal bfloat16 and `w` is normal with std 0.02, so `y` has std about 1.7. Both are built-in distributions; the quantization outputs are
computed from `x` by the operator, so `x` needs no generator.

**Workloads.** M is the token count. Decode batches 1 to 512 and unaligned counts (3, 17, 100, 1000) are correctness-only (M = 512 moves about 20 MB, under the
10 us floor); performance cases are prefill token counts from 1024 to 131072 (128x). Estimated baseline time (bytes / 4.5 TB/s + bf16 flops at 1 PFLOP/s + 4 us): M = 1024
about 14 us (26 MB read and written, 3.8 GFLOP), M = 131072 about 1.9 ms (3.4 GB). The tuned table asks for split-K for M <= 512 (14 partitions for M <= 32, 7 for M <= 128 and 8 up to M = 512
after `compute_splitk_params`) and none above; partials are float32, written to a per-call allocation and added by the reduce kernel in a fixed order, so the result is
deterministic (no atomics). Those cases are correctness-only here.

**Baseline.** AITER's Triton `_fused_gemm_a16w16_quant_x_kernel` (a single 1D grid: the first programs tile the GEMM, the rest quantize one BLOCK_SIZE_M x BLOCK_SIZE_K tile of `x`
each) with `_mxfp8_quant_op`, the split-K reduce kernel, the launcher logic and the N = 256, K = 7168 gfx950 table inlined (see `provenance` in `task.yaml`;
`baseline/LICENSE` is the AITER MIT license). The kernels are unchanged. The upstream files carry the SPDX header `MIT` and `Copyright (C) 2024-2026, Advanced Micro Devices, Inc.`,
kept verbatim; `kernel_repr.py` and `splitk_reduce.py` carry none (the repository MIT LICENSE applies). No third-party origin is noted in any vendored file or in its history.

**Exactness.** The scale of the quantization branch uses `tl.log2` and `tl.exp2` on floats; exactness of `x_quant` (and of `x_scale` at power-of-two amax boundaries) rests on those
being exact for powers of two on gfx950, as for the sibling task.
