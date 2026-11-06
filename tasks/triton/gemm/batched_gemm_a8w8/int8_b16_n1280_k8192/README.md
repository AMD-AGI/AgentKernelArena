# Batched int8 GEMM with per-token and per-channel scales

`out[b] = (x[b] @ w[b]^T) * (x_scale[b] * w_scale[b])` for 16 batch elements, N = 1280, K = 8192. `x`
holds int8 activation codes with one float32 scale per row (`x_scale` [batch, m, 1]); `w` holds int8
weight codes with one scale per output channel (`w_scale` [batch, 1, n]). The integer products are
accumulated exactly, multiplied by the outer product of the scales and rounded to bfloat16. The reference
is the same expression in float64 (exact for the integer sum).

**Inputs.** `inputs.py` quantizes row-wise (scale = row amax / 127, round to nearest, clamp to +-127)
normally distributed activations and weights, so codes and scales are mutually consistent and the
outputs have a realistic magnitude.

**Workloads.** The two old tasks (an AITER Triton extraction and a PyTorch-oracle task) used synthetic
shapes, B = 16, N = 1280, K = 8192, with M from 1 up to a few hundred. No production batched int8
consumer was identified, so those dimensions are kept and M is extended to 8192 so that the large cases
are compute-bound (every performance case is estimated above the 10 us floor; M = 64 is about 14 us).
Smaller M values are correctness-only. M = 3000 exercises the config-table row that has no `M_LEQ`
neighbor.

**Baseline.** AITER's Triton `_batched_gemm_a8w8_kernel` with its gfx950 tuned configuration table (see
`provenance` in `task.yaml`). One deliberate fix: upstream computed the scale offsets as
`pid * BLOCK + arange % M`, whose modulo binds only to the arange, so the tail tile read the scale
tensors out of bounds (the values only fed masked rows). The offsets are parenthesized. `baseline/LICENSE`
is the AITER MIT license.

**GPU audit risks.** The upstream kernel accumulates int8 `tl.dot` results into a float32 accumulator;
partial sums per 64-wide K step are exact, the running float32 sum is not, which the baseline-relative
bound absorbs.
