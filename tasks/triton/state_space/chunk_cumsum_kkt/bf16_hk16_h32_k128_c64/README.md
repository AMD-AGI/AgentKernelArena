# Fused chunk-local gate cumsum + scaled K K^T (Gated DeltaNet prefill), bfloat16, 16 key / 32 value heads, K = 128, chunk 64

The first kernel of AITER's Gated DeltaNet chunked prefill. Tokens are split into chunks of 64 (the last
may be partial). For every value head `h` (key head `h // 2`) and chunk it

1. cumulates the raw per-token log-decay `g` inside the chunk, `gc[t] = sum_{u <= t in chunk} g[u]`, and
   writes it head-major as `g_cumsum[h, t] = gc[t] * log2(e)` (log2 space: the later kernels use `exp2`);
2. writes the strictly lower triangle of the gated, beta-scaled key Gram matrix
   `A[i, h, j] = beta[i, h] * <k[i, h // 2], k[j, h // 2]> * exp(gc[i] - gc[j])`, `i > j` (the diagonal,
   everything above it and the padding rows of a partial chunk are zero).

`k` is `[T, 16, 128]` bfloat16, `beta` and `g` are `[T, 32]` float32, `g_cumsum` is `[32, T]` and `A` is
`[T, 32, 64]` float32. One sequence batch (no variable-length metadata), as in the sibling Gated
DeltaNet tasks of this repository: with `cu_seqlens` AITER builds the chunk schedule on the host
(`prepare_chunk_indices`, `.tolist()`), which a graph-captured call cannot do.

## Baseline

`fused_chunk_local_cumsum_scaled_dot_kkt_fwd_kernel` of
`aiter/ops/triton/_triton_kernels/gated_delta_rule/prefill/fused_cumsum_kkt.py` (ROCm/aiter at
`652f2d4b768239aefd515c929e84817681ff598d`), body unchanged, launched as
`fused_chunk_local_cumsum_scaled_dot_kkt_fwd` does for `chunk_gated_delta_rule_fwd_opt_vk`
(`use_exp2=True`, float32 outputs, no `cu_seqlens`). The kernel loads a `[64, 32]` key tile per K step,
multiplies it by `beta` in float32, **rounds `beta * k` to bfloat16** and accumulates `tl.dot` in float32
(so the baseline's error against an exact evaluation is bfloat16-level, SQNR about 55.6 dB on `A`), then
applies the decay `exp2(gc_i - gc_j)`. The upstream `@triton.autotune` has one entry unless the
environment variable `GATED_DELTA_RULE_TRITON_AUTOTUNE` is set (block `BK` 32, 4 warps, 2 stages); that
entry is passed at launch instead, which also keeps a first call from tuning inside a graph capture.
The older unfused kernel of the same file is not used and removed.

**Provenance note.** The upstream file has no SPDX or copyright header, so the repository `LICENSE` (MIT)
applies and is copied to `baseline/LICENSE`. The other kernels in that directory say "Adapted from
flash-linear-attention: Copyright (c) 2023-2025, Songlin Yang, Yu Zhang" (MIT), and this kernel is of the
same family (block pointers, `chunk_indices` varlen path, `chunk_scaled_dot_kkt` structure), but it
fuses what flash-linear-attention keeps as two kernels and it names no origin; its history in the checkout
starts at "[Triton] Support GDR prefill and decode (#1594)". The flash-linear-attention license is MIT, the
same license as the repository's, so no different license is implied; the exact origin of the
fusion was not established. The file also imports `gated_delta_rule_utils.py`, whose header reads
"Copyright (C) 2023-2026, Songlin Yang, Yu Zhang"; only `log2(e)` and `tl.exp` from it and `utils/op.py`
are reproduced.

## Scenario and config citations

Qwen3-Next-80B-A3B-Instruct. Source: the public `config.json` of Qwen/Qwen3-Next-80B-A3B-Instruct on Hugging Face (`https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/raw/main/config.json`): `linear_conv_kernel_dim` 4, `linear_num_key_heads` 16, `linear_key_head_dim` 128, `linear_num_value_heads` 32, `linear_value_head_dim` 128. The same values are the defaults of `Qwen3NextConfig` (`vllm/transformers_utils/configs/qwen3_next.py` in vLLM at `edbc4969a76b25c531f7b9bb16b79984e65ef023` and `python/sglang/srt/configs/qwen3_next.py` in SGLang), which corroborates them. The chunk size 64 is the one hard-coded in `chunk_gated_delta_rule_fwd_opt_vk` of the AITER file `prefill/chunk.py` (`chunk_size=64`, `BT` 64).

## Inputs

- `k ~ N(0, 1/128)` (the element standard deviation of L2-normalized keys), `beta ~ U(0, 1)` float32
  (`beta = sigmoid(b)` is produced in float32 by the AITER gating kernel `fused_gdn_gating_and_sigmoid`).
- `g` (`inputs.py`): the production formula `-exp(A_log) * softplus(a + dt_bias)` (non-positive, verified in
  that gating kernel) with `a + dt_bias ~ N(0, 1)` and a per-head rate `exp(A_log)` drawn log-uniformly in
  `[0.02, 0.4]`. The per-head rates are a design choice of this task (trained `A_log` values were not
  available): over a 64-token chunk the heads keep between about `exp(-1)` and `exp(-20)` of their state, so
  `gc` reaches about -20 and `g_cumsum` about -29, and the decay is neither trivial nor fully saturated.

## Reference

Float64, per head over groups of 1024 chunks: chunk-local cumulative sum of `g`, scaled by
`log2(e)` for `g_cumsum`, the Gram matrix of the (exactly widened) keys, the natural-log decay
`exp(gc_i - gc_j)` and `beta`, masked to the strict lower triangle; padding rows are zero. It does not use
the kernel's `exp2`/log2-space formulation.

## Comparison

- `A`: `baseline_relative` with `max_violation_fraction: 1.0e-6` (default 0, `k` stays 2). A calibration
  probe (the reference evaluated in float32) violated the default bound on 1.07e-8 of the elements
  at 409600 tokens (peak 2.56x): an element where the baseline's bfloat16-level error happens to be tiny
  next to a float32 evaluation's rounding. The CPU reproduction with the probe's own float32 evaluation, against
  baseline emulations (cumsum as a scan or sequential), gives 0 to 5.96e-8 of the elements (0 to 2 elements)
  with peaks 0.6x to 2.3x (16384 tokens, two draws; 65536 tokens, one draw: 0 and 1.5e-8). The limit
  1e-6 is about 17x above the worst CPU draw and 90x above the GPU observation, and the peaks stay far under
  the comparator's 16x admission limit. The noop, zeros, half-written and stale controls violate on about half
  of the nonzero lower triangle or more, and fp8 rounding on 0.43 of `A` (peaks above 1e4x), so they stay
  rejected; the 16-ulp shift control is rejected through `g_cumsum` as described below.
- `A` (analysis behind the default bound): the default `baseline_relative` bound. A CPU float32 emulation of the baseline (bfloat16 `beta * k`,
  float32 accumulation, `exp2` decay) against the float64 reference gives 55.6 dB; every float32-operand
  design tried passes it with a peak of at most 0.20x the bound at 1000 tokens (two draws): `beta` applied
  to the keys in float32 or after the Gram product, the decay as `exp2` of the log2 gate difference,
  `exp` of the natural one, or `exp2(l_i) * exp2(-l_j)`, and a different cumsum order. One design that
  rounds `beta` to bfloat16 before scaling the keys (the same error magnitude as the baseline, but
  independent of it) fails at 1.9e-2 of the elements with peaks of 13x to 18x; this is the intended
  meaning of the bound (as accurate as the baseline, not a different bfloat16-grade design). `beta` is
  float32 and continuous here, so there are no rows where `beta * k` is exact in bfloat16.
- `g_cumsum`: `baseline_relative` with `k: 8` (the default is 2). A float32 cumulative sum over 64 elements
  has an error that depends on the summation order, and `|gc|` reaches 20 to 30. A CPU float32 study of
  three orders (sequential, 8 x 8 blocked, and a Kogge-Stone scan like a GPU scan) used pairwise as
  baseline and candidate (65, 1000, 4096 tokens with 25 draws each; 16384 tokens with 3 draws) violates the
  default bound for every pair of different orders (e.g. blocked against scan: 3.2e-3 of the elements at
  4096 tokens, peak 1.7x) and still at `k = 4` (up to 4.1e-4 of the elements, peaks 1.1x to 1.5x). At `k = 8` no
  pair violates in any draw (peak at most 0.94x, typically 0.4x to 0.85x). The evaluator's 16-ulp shift
  control, applied to `g_cumsum`, violates at `k = 8` on 0.68 to 0.80 of the elements with peaks of 4.0x to
  5.7x, and rounding to fp8 is far outside, so the controls stay rejected. (The shift control cannot be
  rejected through `A`: its baseline error is bfloat16-level, 1e-3, against a 1e-6 shift; `g_cumsum` is
  what rejects it, because a control must pass every output to pass.)

## Workloads

Token counts from 4096 to 409600 (100x span). Traffic is about 12.7 KB per token (4 KB of keys, 0.25 KB of
`beta` and `g`, 8 KB of `A`, 0.13 KB of `g_cumsum`), so the smallest performance case moves about 52 MB and
is estimated at 15.6 us on MI355X (52 MB / 4.5e12 B/s + 4 us); the largest moves 5.2 GB (about 1.2 ms).
Production prefill chunks are typically 2048 to 16384 tokens per step; larger counts stand for batched or
long-context prefill and are needed for the 100x span. Axis minimum 1, a partial chunk (65), 100
(non-power-of-two, not a multiple of 16) and 1000 are correctness-only. The largest
case needs about 1.7 GB of keys plus the 6.7 GB float64 `A`.
