# Gated RMSNorm, bfloat16, head size 128, SiLU gate

`out = x * rsqrt(mean(x^2) + epsilon) * weight * silu(z)` along the last axis, with the normalization applied
before the gate (`silu(z) = z * sigmoid(z)`), no bias and no mean subtraction. This is the output norm of the gated
delta net (GDN) linear attention of Qwen3-Next style models: every (token, value head) pair is one row of 128
features, `z` is the gate projection of the same shape, `weight` is shared by all heads and `epsilon` is 1e-6. The
reference works in float64 in chunks of 262144 rows; the output is rounded to bfloat16 and compared with the default
baseline-relative bound.

## Baseline

vLLM's Triton `_layer_norm_fwd_1pass_kernel` (the gated LayerNorm / RMSNorm of the mamba and flash-linear-attention
code), vLLM commit `edbc4969a76b25c531f7b9bb16b79984e65ef023`, kernel body unchanged: one program per row, a single
128-wide block, one warp. The upstream launcher is reduced to this scenario (RMS mode, no bias, SiLU gate after the
norm, group size equal to the row width) and `run()` allocates the per-row reciprocal standard deviation that the
kernel stores as a side output. The LayerNorm, sigmoid-gate, bias, gate-before-norm and grouped variants of the
kernel, which four earlier tasks covered (vLLM mamba and FLA copies, an SGLang fused norm-gate), are flags dropped
from this scenario. The mamba copy carries Tri Dao's copyright line, kept in the file header.

## Workloads

A row moves 768 bytes (x, z and out), so a case needs about 35k rows to exceed the 10 us timing floor; performance
cases start at 65536 rows (a 2048-token chunk with 32 value heads) and span 100x up to 6.55M rows (5 GB moved). The
largest case needs about 15 GB for the float64 reference. Smaller row counts, from a single row up, are
correctness-only.

## GPU audit risks

The smallest performance case is estimated at about 15 us, close to the floor; the one-warp-per-row baseline has
millions of tiny programs at the large end.
