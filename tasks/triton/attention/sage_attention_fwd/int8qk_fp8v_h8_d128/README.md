# SageAttention v1, int8 Q/K with fp8 V, causal, 8 heads of 128 channels

SageAttention v1 for diffusion transformers: bfloat16 `q`, `k`, `v` in `[batch, seqlen, heads, head_dim]`
layout go in, bfloat16 attention output comes out. **The quantization is part of the operator and of
the timed call**: K is smoothed (mean over the sequence subtracted), Q and smoothed K are quantized to
int8 per block (256 query rows / 128 key rows, with the softmax scale and log2(e) folded into Q), V is
quantized to float8 e4m3 per head channel, and the flash-attention kernel runs int8 QK and fp8 PV
products with the descales fused into a base-2 online softmax. Self-attention is causal.

**Production origin.** AITER's `fav3_sage` (`aiter/ops/triton/attention/fav3_sage.py`) with the
`sage_attention_quant` kernels, as used for video and image diffusion (Wan, HunyuanVideo, Qwen-Image
edit); the old task only measured 64-256 tokens, the workloads here are 1k-16k tokens per sequence.
`n_heads = 8` keeps the memory of the batch-8 cases moderate; query and KV heads are equal (MHA).

**Baseline.** AITER's Triton kernels vendored at the pinned commit (MIT, `baseline/LICENSE`). Two
changes were needed to satisfy the candidate rules for a Triton baseline:

* The AITER wrapper computes `k.mean`, `k - k_mean` and `v.abs().amax` with PyTorch. They moved into
  Triton: the new `_sage_kv_stats_kernel` produces the K mean and the V channel scales, and
  `sage_quant_kernel` subtracts the mean while quantizing K (rounded to bfloat16 like the original
  `k - k_mean`). This kernel is written for this task, not taken from AITER, and runs one program
  per (batch, head, 32-channel slice), which is slow for long sequences.
* `fav3_sage` + `sage_quant` + `fav3_sage_func` were merged into `run()` with the gfx950
  configuration of `get_sage_fwd_configs` (256x128 tiles, 8 warps, 3 stages) and no LSE output; the
  output is written into the preallocated tensor instead of a zero-initialized allocation.

**Reference and comparison.** float64 un-quantized causal attention (query-chunked). Quantization
noise is large (about 29 dB SQNR in a CPU emulation of the pipeline for Gaussian inputs, against the
baseline sanity floor of 25 dB), so the comparison departs from the default bound:
`k: 3, max_violation_fraction: 0.001`. With the default `k = 2`, an implementation that quantizes in
a different tiling (for example 128x64 blocks, or without K smoothing) violates the bound on about 1% of
elements although its SQNR is the same as the baseline's; at `k = 3` such variants stay below 0.04%,
while rounding the output to fp8 (0.25%) or shifting it by 16 output ulps (6%) is still rejected.
Elements remain bounded by three times the baseline's own error or its per-row RMS.

**Scope.** The retired task also covered 64-channel heads, grouped-query heads, non-causal attention and a
causal sliding window at 64-256 tokens. This scenario is the causal, equal-head, 128-channel configuration
that diffusion transformers use; the baseline kernel keeps those branches as constexpr-off code but
`run()` only exercises the causal path, so they are not part of this task's contract.

**Workloads.** `batch` 1-8, `seqlen` 1k-16k for performance (128x span of `batch * seqlen`),
shorter and unaligned lengths (64, 100, 129, 257, 333, 1000, 4100) are correctness-only and cover
partial quantization blocks. Inputs are standard normal (Q, K and V independently).

**Risks for the GPU audit.** Shortest sequences (`seqlen = 64`, `100`) are smaller than one
256-row query block, which the original harness (64-row blocks) never exercised. The smallest
performance case (`1 x 1024`) runs three kernels (statistics, quantization, attention) and is
expected to be above 10 us. The SQNR margin over the 25 dB baseline sanity floor is about 4 dB
(emulated, not measured on the GPU).
