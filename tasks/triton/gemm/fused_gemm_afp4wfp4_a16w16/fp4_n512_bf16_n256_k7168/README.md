# Fused MXFP4 GEMM + bf16 GEMM, DeepSeek-R1 layer input (N = 512 + 256, K = 7168)

One launch computes two GEMMs over the same tokens:

- `y_a4w4 = dequant(x_fp4) @ dequant(w_fp4)^T`, an MXFP4 x MXFP4 GEMM (e2m1 codes, two per byte with the even K index in the
  low nibble, e8m0 scales per 32 values of K) with N = 512: the gate-up projection of the shared expert.
- `y_a16w16 = x_bf16 @ w_bf16^T`, a bf16 GEMM with N = 256: the MoE router gate.

Both accumulate in float32 and round to bfloat16 (no bias). The operator is AITER's `fused_gemm_afp4wfp4_a16w16`; its docstring
names the use: "fusing the gate up-projections and MOE gating". The plain (non-preshuffled) weight layout is the interface; the
preshuffled-weight variant that is the wrapper's default is not used (the same choice as the sibling task
`tasks/triton/gemm/gemm_afp4wfp4`).

**Scenario and config citations.** DeepSeek-V3 `config.json` (Hugging Face; DeepSeek-R1 has the same architecture fields):
`hidden_size` 7168 (K), `n_routed_experts` 256 (N of the router gate), `n_shared_experts` 1 and `moe_intermediate_size` 2048
(a shared expert of width 2048). The shared expert width is sharded over eight GPUs (tensor parallelism 8) as 2048 / 8 = 256
per rank, so the gate-up projection has 2 x 256 = 512 output channels. TP = 8 is a deployment choice, not a config field; it
is the (N4, N16, K) = (512, 256, 7168) shape that AITER's benchmark
(`op_tests/op_benchmarks/triton/bench_fused_gemm_afp4wfp4_a16w16.py`) uses by default and for which AITER ships the tuned table
inlined in the baseline. The MXFP4 weight format is that of an MXFP4-quantized DeepSeek-R1 checkpoint as served through AITER
(SGLang `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py`, which calls AITER's afp4wfp4 kernels). The
pinned vLLM and SGLang trees contain no call site of this fused operator; the model layer is identified from the shape, the
docstring and AITER's tuned file only.

**Reference.** `reference.py` dequantizes both MXFP4 operands from codes and scales (the e2m1 grid 0, 0.5, 1, 1.5, 2, 3, 4, 6 and
the sign bit; scale 2^(s - 127)) and multiplies in float64, row-chunked; the bf16 GEMM multiplies the bf16 operands in float64.
It is independent of the baseline and of the quantizer in `inputs.py`.

**Inputs.** `x_bf16` is standard normal and `x_fp4` is its MXFP4 quantization (like AITER's dynamic MXFP4 quantization: the
scale of a 1x32 group is the group maximum rounded to a power of two minus the exponent of the largest e2m1 value, codes
round to nearest with ties to the even code, saturating at 6), so both GEMMs see the same hidden states, as in the layer.
`w_fp4` quantizes a normal weight (std 0.02) with the same rule; `w_bf16` is normal with std 0.02. Activation quantization is
a separate upstream kernel in production, so it is input here, not timed.

**Workloads.** M is the token count of the layer input. Decode batches 1 to 512 and unaligned counts (3, 17, 100, 1000) are
correctness-only (M = 512 moves about 16 MB and estimates to about 10 us, at the floor, so it is not scored); performance cases are prefill token counts from 1024 to
131072 (128x). Estimated baseline time (bytes / 4.5 TB/s + flops at 4 PFLOP/s fp4 and 1 PFLOP/s bf16 + 4 us): M = 1024 about
15 us, about 1.5x above the 10 us floor (26 MB, 3.8 GFLOP fp4 and 3.8 GFLOP bf16 at 1 PFLOP/s), M = 131072 about 1.7 ms.

**Baseline.** AITER's Triton `_fused_gemm_afp4wfp4_a16w16_kernel` (`tl.dot_scaled` e2m1 x e2m1 for the first N programs, a
bf16 `tl.dot` for the rest) with its reduce kernel, the launcher logic and the shape-specialized gfx950 table
(`configs/gemm/gfx950-FUSED-GEMM-AFP4WFP4-A16W16-N4=512-N16=256-K=7168.json`) inlined (see `provenance` in `task.yaml`;
`baseline/LICENSE` is the AITER MIT license). The kernels are unchanged. The table asks for split-K for M <= 128 (14 partitions
for M <= 8, 7 for M <= 64 and 4 for M <= 128 after `get_splitk`) and none from M = 256; partial sums are float32, written to a per-call allocation
and added in a fixed order by the reduce kernel, so the result is deterministic (no atomics). All split-K cases are
correctness-only here. The launcher's optional ahead-of-time compiled metadata is not used. The upstream files carry the
SPDX header `MIT` and `Copyright (C) 2024-2026, Advanced Micro Devices, Inc.`, kept verbatim; no third-party origin is noted
in any vendored file or in its history. `tl.dot_scaled` needs gfx950; the tile rows with BLOCK_SIZE_N 16 are raised to 32 by the launcher.
