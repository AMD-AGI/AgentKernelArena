# Dense MXFP8 GEMM, N = 1280, K = 6144

`out = (x * x_scale) @ (w * w_scale)^T`: float8_e4m3fn activations `x` [m, 6144] and weights `w` [1280, 6144], each
carrying one e8m0 scale byte per 1x32 group along K (OCP microscaling MXFP8; byte `b` encodes `2^(b - 127)`). The
product is accumulated in float32 and rounded to bfloat16. N = 1280, K = 6144 is the per-rank (TP=8) `qkv_proj` of
MiniMax-M3-MXFP8. The reference dequantizes both operands exactly in float64 (an fp8 value times a power of two is
exact) and multiplies in float64, in row chunks of 8192.

**Timed region.** The GEMM only. The per-token activation quantization is a separate kernel in production and is
not part of this operator: `x` and `x_scale` are inputs, as in the profiled leaf of the old task.

**Inputs.** `inputs.py` draws the activation as N(0, 0.5^2) and the weight as N(0, 0.1^2) in bfloat16 and quantizes
them with the serving quantizer's semantics (scale exponent `ceil(log2(amax / 448)) + 127`, RNE to float8_e4m3fn), so
codes and scales are mutually consistent and span realistic magnitudes.

**Workloads.** The old task had decode M = 1 and 64 and prefill M = 16384 for this family (and an `o_proj` family,
N = 6144, K = 1024, which is the sibling scenario `n6144_k1024`). Performance cases run from 1024 to 131072 tokens
(128x); M = 1024 is estimated at about 12 us (fp8 flops / 2e15 plus the 4 us floor), so M <= 512 is correctness-only.
The largest case holds 805 MB of activations and 335 MB of output. The old tolerance was a 6% relative-norm gate; the
default baseline-relative bound replaces it.

**Baseline.** SGLang's `_mxfp8_linear_kernel` (`tl.dot_scaled`, 64x{64,128}x128 tiles, 4 or 8 warps as in
`_run_mxfp8_linear_kernel`), vendored from the checkout recorded in `provenance`. In that revision the file moved
from `srt/layers/quantization/` to `kernels/ops/quantization/`; the kernel body and launcher tile selection match the
old task's harness expectations (the old pinned revision was not available to diff byte for byte). The folded-in
rocmbench `test_matmul_MXFP` (a `tl.dot_scaled` unit test with its own conversion kernel) adds no scenario beyond this
operator and is not carried over. `tl.dot_scaled` exists only on gfx950.

**GPU audit risks.** Lowest performance case (M = 1024) is near the 10 us floor; raise it if the audit measures less.
`run()` does no PyTorch compute (scalars, strides and a kernel launch only), so the baseline satisfies the Triton
candidate rules.
