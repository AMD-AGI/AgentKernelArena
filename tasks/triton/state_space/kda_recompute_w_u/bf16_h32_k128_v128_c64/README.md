# KDA recompute of w, u and kg, bf16, 32 heads, K = V = 128, chunk 64

The WY-representation recompute of Kimi Delta Attention (Kimi-Linear) chunked prefill. For every
chunk of 64 tokens and every head, with `A` the chunk's lower-triangular inverse (the output of the
triangular solve), the write strength `beta` and the cumulative log2-decay `gk`:

```text
u  = A @ (beta * v)
w  = A @ (beta * k * 2^gk)
kg = k * 2^(gk[last] - gk)        (last = last valid token of the chunk)
```

`k` is `[T, 32, 128]`, `v` `[T, 32, 128]`, `beta` `[T, 32]`, `A` `[T, 32, 64]` (all bfloat16) and
`gk` `[T, 32, 128]` float32; `w`, `u` and `kg` are bfloat16 with the shapes of `k`, `v` and `k`.
One sequence batch (no variable-length metadata).

- **Origin.** `recompute_w_u_fwd` of flash-linear-attention as carried in vLLM
  (`vllm/third_party/flash_linear_attention/ops/kda.py`). The chunked forward calls it with the gates,
  so it also writes the decayed keys `kg` that the state-update kernel consumes; the earlier task
  left that output out, here it is part of the operator. Dimensions follow Kimi-Linear (32 heads,
  head size 128, chunk 64).
- **Inputs.** `k ~ N(0, 1/128)`, `v ~ N(0, 1)`, `beta ~ U(0, 1)`; `A` and `gk` come from `inputs.py`:
  `gk` as the cumulative log2-decay gates (per-token decays in `[-0.25, 0]` cumulated inside each
  chunk, non-increasing), and `A` as the exact inverse of `I + L` for a random strictly lower `L`
  with N(0, 0.05^2) entries (the size of the beta-scaled key products), confined to the valid tokens
  so the inverse is unit lower triangular and zero above the diagonal, as the solve produces.
- **Baseline.** The vLLM kernel for one sequence (varlen path, batch axis and the unused `qg`
  output removed), autotune kept, block 64 over keys and values, `ieee` dot precision for `u`.
  `beta * v`, `beta * k` and the decayed operands are rounded to bfloat16 before the MFMA dots as in
  vLLM, so the baseline's error against the float64 reference is that of production; the default
  `baseline_relative` bound is derived from it and the framework checks the baseline's SQNR.
- **Workloads.** 1024 to 131072 tokens (128x span). Traffic is about 52 KB per token (k and v 8 KB
  each, gk 16 KB, A 4 KB, w, u and kg 8 KB each), so the smallest case is about 53 MB and should
  take the baseline over 10 us. Production prefill chunks are typically 2048-16384 tokens per step;
  larger counts stand for batched or long-context prefill and are needed for the span. Axis
  minimum 1, 17, 65 (a partial chunk) and 1000 (non-power-of-two, not a multiple of 16) are
  correctness-only.
- **Comparison.** Default `baseline_relative` on `w`, `u` and `kg`. The reference is float64 (about
  4.3 GB per output at the largest case, inputs about 5 GB).
