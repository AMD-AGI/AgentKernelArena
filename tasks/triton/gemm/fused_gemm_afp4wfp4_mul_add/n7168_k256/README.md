# Fused MXFP4 GEMM + mul-add, DeepSeek-R1 shared-expert down projection (N = 7168, K = 256)

`out = a * b + dequant(x) @ dequant(w)^T`: an MXFP4 x MXFP4 GEMM (e2m1 codes, two per byte with the even K index in the low
nibble, e8m0 scales per 32 values of K) whose epilogue multiplies the tensor `b` [m, 7168] by the scalar `a` and adds it to the
accumulator, with a single rounding to bfloat16. This is AITER's `fused_gemm_afp4wfp4_mul_add` with `fuse_type=1`
(`a * b + Y`), on the plain (non-preshuffled) weight layout like the sibling task `tasks/triton/gemm/gemm_afp4wfp4`.

**Production use.** In a DeepSeek MoE layer the shared expert output `Y` is summed with the routed experts' output scaled by
`routed_scaling_factor`: SGLang's `python/sglang/srt/models/deepseek_v2.py` (pinned commit) does
`x = shared_output; x.add_(final_hidden_states, alpha=self.routed_scaling_factor)`. With `Y` as the GEMM, `a` =
`routed_scaling_factor` and `b` = the routed output, this is exactly `fuse_type=1`. AITER's tests and benchmark for this kernel
(`op_tests/triton_tests/gemm/fused/test_fused_gemm_afp4wfp4_mul_add.py`) use N = 7168, K = 256 and do not name a model; the pinned
SGLang and vLLM trees do not call the fused operator, so the correspondence to the layer is by shape and semantics.

**Scenario and config citations.** DeepSeek-V3 `config.json` (Hugging Face; DeepSeek-R1 has the same architecture fields):
`hidden_size` 7168 (N), `moe_intermediate_size` 2048 with `n_shared_experts` 1 (shared expert width 2048) and
`routed_scaling_factor` 2.5 (the scalar `a`, the task's default). Under tensor parallelism 8 the shared expert width per rank is
2048 / 8 = 256, which is K of the down projection. TP = 8 is a deployment choice, not a config field; it matches the
(N, K) = (7168, 256) that AITER's tests, benchmark and tuned files use. The MXFP4 weight format is that of an MXFP4-quantized
DeepSeek-R1 checkpoint as served through AITER.

**Reference.** `reference.py` dequantizes both operands from codes and scales (the e2m1 grid 0, 0.5, 1, 1.5, 2, 3, 4, 6 and the
sign bit; scale 2^(s - 127)), multiplies in float64 (row-chunked) and returns `a * b + product` in float64. It is independent
of the baseline and of the quantizer in `inputs.py`.

**Inputs.** `inputs.py` quantizes standard normal activations and normal weights (std 0.06) with the rule of AITER's dynamic MXFP4
quantization (scale of a 1x32 group = group maximum rounded to a power of two minus the exponent of the largest e2m1 value,
codes round to nearest with ties to the even code, saturating at 6). The product has std about 1; `b` is normal with std 0.4, so
`a * b` has std 1.0 and neither term dominates the bfloat16 output rounding. The magnitudes are chosen for sensitivity, they are
not measured from the model. `a` is the scalar 2.5.

**Workloads.** M is the token count. Decode batches and chunks up to 1000 tokens (including the upstream test sizes 1, 8 and 31) are
correctness-only; performance cases are prefill token counts from 1024 to 131072 (128x). The kernel is bandwidth-bound: per token it
reads `b` (14 KB) and writes `out` (14 KB). Estimated baseline time (bytes / 4.5 TB/s + fp4 flops at 4 PFLOP/s + 4 us): M = 1024
about 12 us, about 1.2x above the 10 us floor (31 MB), M = 131072 about 1.7 ms (3.8 GB).

**Baseline.** AITER's Triton `_fused_gemm_afp4wfp4_mul_add_kernel` (`tl.dot_scaled` e2m1 x e2m1) with its reduce kernel, the launcher
logic and `get_splitk` and the table inlined (see `provenance` in `task.yaml`; `baseline/LICENSE` is the AITER MIT license). The table is
the gfx950 FUSED-GEMM-AFP4WFP4-MUL_ADD family file for N = 7168, K = 256 (`configs/gemm/gfx950-FUSED-GEMM-AFP4WFP4-MUL_ADD-N=7168-K=256.json`;
file names use the unpacked K); every row has NUM_KSPLIT 1, so the reduce kernel is never launched here and there is no split-K
nondeterminism. The kernels are unchanged. The upstream files carry the SPDX header `MIT` and `Copyright (C) 2024-2026, Advanced Micro
Devices, Inc.`, kept verbatim; no third-party origin is noted in any vendored file or in its history.

**Scalar argument.** `a` is passed as a Python float, which the launcher treats as a scalar operand (`IS_A_SCALAR`); `b` is a
contiguous [m, 7168] tensor (`IS_B_TENSOR`).

**Kernel shape.** `tl.dot_scaled` needs gfx950. K = 256 is a single BLOCK_SIZE_K = 256 step, so the kernel is almost purely a
bandwidth kernel.
