# SageAttention v2 with MXFP4 Q/K, causal, 8 heads of 128 channels

SageAttention v2 for diffusion transformers: bfloat16 `q`, `k`, `v` in `[batch, seqlen, heads, head_dim]`
layout go in, bfloat16 attention output comes out. **The quantization is part of the operator and of the
timed call**: Q and K are Hadamard-rotated (block 64), K is smoothed (mean over the sequence subtracted
after rotation), Q is smoothed per block of 256 query rows (the block mean enters the attention kernel as
a score bias), both are quantized to MXFP4 (E2M1 values, one E8M0 scale per 32 channels), V is quantized
to float8 e4m3 with one scale per head channel, and the attention kernel multiplies with
`tl.dot_scaled` and FP8 probabilities. Self-attention is causal. The result approximates
`softmax(q k^T / sqrt(128), causal) v`.

**Production origin.** AITER's `fav3_sage_attention_mxfp4` (`aiter/ops/triton`), the MXFP4 path of the
SageAttention v2 family used for video and image diffusion. The previous task only measured 128-384
tokens; the workloads here are 512 to 8192 tokens per sequence. `num_heads = 8` keeps the batch-8 cases
moderate in memory; query and KV heads are equal (MHA). gfx950 only (E2M1 `dot_scaled`).

**Baseline.** AITER's Triton kernels vendored at the pinned commit (MIT, `baseline/LICENSE`), specialized
to causal bshd attention with Q smoothing, Hadamard block 64, K smoothing, no LSE and no block-sparse
mode. Changes, all needed to satisfy the candidate rules for a Triton baseline or carried over from the
previous task's numerical repairs:

* The AITER wrapper computes the Hadamard matrix, the V abs-max, the K mean and subtraction and the
  concatenation of the expansion components with PyTorch. They are Triton kernels here
  (`_hadamard_kernel`, `_v_scale_kernel`, `_k_mean_kernel`, `_k_sub_mean_kernel`); these four kernels are
  written for this task, not taken from AITER, and are simple rather than fast.
* The quantizers use nearest-even E2M1 conversion with OCP power-of-two scales and a zero-safe V
  quantizer. Q and K are quantized as a two-component MXFP4 expansion (the quantized residual of the first
  component is the second), evaluated as four feature products `(Q0+Q1)(K0+K1)^T`; the MXFP4 downcast
  kernel writes each component into two slots of the expanded operand instead of concatenating.

**Scope.** The previous task also had a non-causal case and grouped-query cases (8 query heads over 2 KV
heads, 4 heads); this scenario fixes causal MHA with 8 heads, the diffusion-transformer use. Non-causal
and GQA attention would be separate scenarios of this operator and are not covered here.

**Reference and comparison.** float64 un-quantized causal attention (query-chunked), returned as float32.
MXFP4 quantization noise is part of the operator, so the comparison departs from the default bound:
`k: 4, max_violation_fraction: 0.001`, still bounded by four times the baseline's own error or per-row
RMS, and the baseline must reach the 25 dB SQNR sanity floor against the reference.

**Workloads.** `batch` 1-8 and `seqlen` 512-8192 for performance (128x span of `batch * seqlen` from
512 to 65536 tokens); short, unaligned and multi-block lengths (1, 17, 128, 256, 300, 384, 1000) are
correctness-only and cover partial 256-row and 128-key blocks. Inputs are standard normal.

**Risks for the GPU audit.** The tuned `(256 x 128, 8 warps, 3 stages)` configuration was chosen for
gfx950 and not re-measured here. The small performance cases run about a dozen kernels, so they should be
above 10 us but this is unmeasured. The SQNR margin over the 25 dB floor and the `k = 4` bound were
inherited from the previous task's calibration and need confirmation on the GPU (the old gate was
normalized max error <= 0.25 with at most 1.5% outliers).
