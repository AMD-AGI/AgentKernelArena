# WY recompute of w and u (Gated DeltaNet), bf16, 16 key / 32 value heads, K = V = 128, chunk 64

The step of Gated DeltaNet chunked prefill (Qwen3-Next / Kimi-Linear style) that rebuilds the WY
representation from the solved intra-chunk inverse. For every chunk of 64 tokens and value head `h`
(key head `h // 2`), with `A[:, h]` the chunk's 64 x 64 inverse of `I + tril(beta K K^T decay)`:

```text
u = A @ (beta * v)
w = A @ (beta * exp(g) * k)
```

`k` is `[T, 16, 128]`, `v` `[T, 32, 128]`, `beta` `[T, 32]` (all bfloat16), `g` `[T, 32]` float32
(chunk-local cumulative log-decay), `A` `[T, 32, 64]` bfloat16 (row `i` of the chunk's block in
token row `i`, zeros beyond the valid rows and columns of a partial last chunk). `w` and `u` are
bfloat16 `[T, 32, 128]`. One sequence batch (no variable-length metadata).

- **Origin.** `recompute_w_u_fwd` of flash-linear-attention as carried in vLLM
  (`vllm/third_party/flash_linear_attention/ops/wy_fast.py`). It replaces two earlier tasks (an
  SGLang copy and a vLLM copy with 8-16 head test shapes); dimensions follow Qwen3-Next's Gated
  DeltaNet (16 key heads, 32 value heads, K = V = 128, chunk 64).
- **Inputs.** `k ~ N(0, 1/128)`, `v ~ N(0, 1)`, `beta ~ U(0, 1)` (a sigmoid output). `inputs.py`
  draws per-token log-decays in `[-0.2, 0]` and cumulates them inside every 64-token chunk (so `g`
  is non-increasing, as in production), then builds `A` as the true inverse of
  `I + tril(beta_i <k_i, k_j> exp(g_i - g_j))` from the generated `k`, `beta` and `g` with a float32
  triangular solve, rounded to bfloat16. `A` is therefore unit lower triangular with realistic
  magnitudes, not random numbers.
- **Baseline.** The vLLM kernel reduced to this scenario: the varlen path and the autotune were
  removed (4 warps, 3 stages, fixed 64-wide K and V blocks, which also keeps a first call from
  tuning inside a graph capture). As in vLLM the beta-scaled (and gated) right operands are rounded
  to bfloat16 before the dot, which accumulates in float32, so the baseline's error against the
  float64 reference is larger than a float32-operand implementation's; the default
  `baseline_relative` bound is derived from that error, and the baseline's SQNR against the
  reference is checked by the framework.
- **Workloads.** 2048 to 204800 tokens (100x span). Traffic is about 32 KB per token (16 KB read,
  16 KB written), so the smallest case is about 65 MB and should take the baseline over 10 us.
  Production prefill chunks are typically 2048-16384 tokens per step; larger counts stand for
  batched or long-context prefill and are needed for the 100x span. Axis minimum 1, a partial chunk
  (65) and 1000 (non-power-of-two, not a multiple of 16) are correctness-only.
- **Comparison.** Default `baseline_relative` for `w` and `u`; the reference runs in float64
  (outputs about 6.7 GB each at the largest case, inputs about 6.6 GB). The CPU pre-check does not
  smoke that case; the generator and reference were checked on 1000 tokens against a per-chunk
  loop and the inverse property of `A`.
