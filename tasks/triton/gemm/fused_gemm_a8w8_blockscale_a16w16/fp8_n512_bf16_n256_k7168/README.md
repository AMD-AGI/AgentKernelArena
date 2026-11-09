# Fused block-scaled fp8 GEMM + bf16 GEMM, DeepSeek-V3 layer input (N = 512 + 256, K = 7168)

One launch computes two GEMMs over the same tokens:

- `y_a8w8 = dequant(x_fp8) @ dequant(w_fp8)^T`, a block-scaled fp8 GEMM (1x128 activation scales, 128x128 weight scales) with
  N = 512: the gate-up projection of the shared expert.
- `y_a16w16 = x_bf16 @ w_bf16^T`, a bf16 GEMM with N = 256: the MoE router gate.

Both accumulate in float32 and round to bfloat16 (no bias, no activation). The operator is AITER's
`fused_gemm_a8w8_blockscale_a16w16`; its docstring names the use: "fusing the gate up-projections and MOE gating".

**Scenario and config citations.** DeepSeek-V3 `config.json` (Hugging Face; DeepSeek-R1 has the same architecture fields):
`hidden_size` 7168 (K), `n_routed_experts` 256 (N of the router gate), `n_shared_experts` 1 and `moe_intermediate_size` 2048
(a shared expert of width 2048), and `quantization_config` with `quant_method` fp8, `fmt` e4m3 and `weight_block_size`
[128, 128] (the 1x128 / 128x128 scales). The shared expert width is sharded over eight GPUs (tensor parallelism 8) as
2048 / 8 = 256 per rank, so the gate-up projection has 2 x 256 = 512 output channels. TP = 8 is a deployment choice, not a
config field; it is the (N8, N16, K) = (512, 256, 7168) shape that AITER's benchmark
(`op_tests/op_benchmarks/triton/bench_fused_gemm_a8w8_blockscale_a16w16.py`) uses by default and for which AITER ships the tuned
table inlined in the baseline. The pinned vLLM and SGLang trees contain no call site of this fused operator; the model layer
is identified from the shape, the docstring and AITER's tuned file only.

**Reference.** `reference.py` dequantizes the fp8 operands from codes and scales and multiplies in float64, row-chunked; the
bf16 GEMM multiplies the bf16 operands in float64. It shares nothing with the baseline.

**Inputs.** `x_bf16` is standard normal; `x_fp8` is its 1x128 block quantization (scale = block amax / 448), so the two GEMMs
see the same hidden states, as in the layer. `w_fp8` is a normal weight (std 0.02) quantized in 128x128 blocks with the same
rule; `w_bf16` is a normal weight with std 0.02. Activation quantization is a separate upstream kernel in production, so it
is input here, not timed.

**Workloads.** M is the token count of the layer input. The decode batches up to 256 and the unaligned counts (3, 17, 100,
1000) are correctness-only: M = 256 moves about 12 MB, which estimates to about 8.6 us. Performance cases are token counts
from 512 to 65536 (128x). Estimated baseline time (bytes / 4.5 TB/s + flops at 2 PFLOP/s fp8 and 1 PFLOP/s bf16 + 4 us):
M = 512 about 12 us, about 1.2x above the 10 us floor (19 MB, 3.8 GFLOP fp8 and 1.9 GFLOP bf16), M = 65536 about 0.9 ms. The largest case is bounded by the
inputs (about 1.4 GB of activations) and the float64 reference outputs (0.4 GB).

**Baseline.** AITER's Triton `_fused_gemm_a8w8_blockscale_a16w16_kernel` (one 1D grid whose first programs tile the fp8 GEMM
and the rest the bf16 GEMM) with its reduce kernel, the wrapper logic and the shape-specialized gfx950 table
(`configs/gemm/gfx950-FUSED-GEMM-A8W8_BLOCKSCALE-A16W16-N8=512-N16=256-K=7168.json`) inlined (see `provenance` in `task.yaml`;
`baseline/LICENSE` is the AITER MIT license). The kernel text is unchanged (see the table correction below). The table asks for split-K (up to 14 partitions)
for M <= 256 and no split-K above 256; the partial sums are float32, written to a per-call allocation and added by the reduce
kernel in a fixed order, so the result is deterministic (no atomics). All split-K cases are correctness-only here.
The upstream files carry the SPDX header `MIT` and `Copyright (C) 2024-2026, Advanced Micro Devices, Inc.`, kept verbatim.
No third-party origin is noted in any vendored file or in its history.

**Table correction (upstream config bug).** The `M_LEQ_256` row of the shape-specialized table has `BLOCK_SIZE_K` 256 (with
`NUM_KSPLIT` 4, 64x64 tiles). The fp8 half of the kernel loads one activation scale and one weight scale per `BLOCK_SIZE_K`
tile and advances the scale pointers by `BLOCK_SIZE_K // GROUP_K`, so it needs `BLOCK_SIZE_K == GROUP_K` = 128: with 256 the
second 128-block of every tile is multiplied by the first block's scale and one scale is skipped. A CPU emulation of exactly that
scale application at M = 256 (split-K 4, 1792 K values each) gives 16.1 dB SQNR against the reference, matching the 16.4 dB
measured on MI355X; with `BLOCK_SIZE_K` 128 the same emulation matches the reference (SQNR above 250 dB). The baseline therefore
uses that row with `BLOCK_SIZE_K` 128 (every other field, the tile shape and split count, is as upstream) and the launcher asserts
`GROUP_K == BLOCK_SIZE_K` for every row; all other rows already have 128. The bf16 half does not depend on `GROUP_K`.
