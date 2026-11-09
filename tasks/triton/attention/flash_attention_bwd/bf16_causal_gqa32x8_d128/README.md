# Flash attention backward, causal GQA 32/8, bfloat16, head_dim 128

Backward pass of causal grouped-query attention for training, in the BSHD layout: `q`, `do` are
`[batch, seqlen, 32, 128]`, `k`, `v` are `[batch, seqlen, 8, 128]`, `lse` is `[batch, 32, seqlen]`
float32. Query head `h` reads KV head `h // 4`. The inputs are what the forward pass saves
(`o`, the bfloat16 forward output, and `lse`, the natural-log row log-sum-exp of the scaled, causally
masked scores) plus the upstream gradient `do`. The operator returns `dq`, `dk` and `dv` (bfloat16);
`dk` and `dv` are the sums over the 4 query heads of each KV head. The backward recomputes
`P = exp(sm_scale * q k^T - lse)`, uses `delta = rowsum(do * o)` and forms
`dS = P * (do v^T - delta)`.

## Production origin of the baseline

AITER's Triton MHA backward, `flash_attn_onekernel_backward` with the config
`bkwd_onekernel` of `gfx950-MHA-DEFAULT.json`: a delta preprocess kernel
(`_bwd_preprocess`) and one fused kernel (`bwd_kernel_causal`) that, per (KV head, key block),
accumulates `dk`/`dv` over all query blocks of the group and in the same program the `dq` rows,
without atomics. The result is deterministic. Source: `ROCm/aiter` at
`652f2d4b768239aefd515c929e84817681ff598d`, files `aiter/ops/triton/_triton_kernels/attention/mha_onekernel_bwd.py`
and `aiter/ops/triton/attention/mha_onekernel_bwd.py` (both carry
`SPDX-License-Identifier: MIT` and `Copyright (C) 2024-2026, Advanced Micro Devices, Inc.`, kept verbatim),
`aiter/ops/triton/utils/_triton/kernel_repr.py` (no header upstream; the repository MIT license applies),
`aiter/ops/triton/utils/_triton/mha_kernel_utils.py` (`SPDX-License-Identifier: MIT`,
`Copyright (C) 2024-2025, Advanced Micro Devices, Inc.`), `aiter/ops/triton/utils/types.py` (no header;
only `_is_fp8` is reproduced) and `aiter/ops/triton/configs/gfx950-MHA-DEFAULT.json`. `baseline/LICENSE`
is the AITER MIT license.

Modifications: see `task.yaml`. In short, the kernels are unchanged except that the always-false
`DEBUG_TRITON` print blocks were deleted (printing is forbidden in a candidate), the config file read
became a literal, and the logger and the architecture probe were removed. `run()` fixes `causal=True`
and disables dropout, ALiBi, sink and sliding window, the options this scenario does not use.

### Relation to AITER's `flash_attn_triton_amd` backward

The onekernel files were added to AITER in commit `b37ca0fa7` ("[TRITON]: Mha fused backward kernels") and carry
AITER's MIT header, which is the license of this baseline. Their structure, function names and comments
(`_bwd_preprocess`, `_bwd_dkdv_inner`, "qT_ptrs = (1, BLOCK_M) + (HEAD_DIM, 1), transpose of q", ...) are shared
with `aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/bwd.py` of the same commit, which has no license header
and is not vendored here. `flash_attn_triton_amd` is the AMD Triton backend that flash-attention also ships; the
ROCm documentation says that flash-attention's AMD Triton backend is now provided by AITER.

## Scenario

bfloat16, 32 query heads over 8 KV heads, head_dim 128, causal, `sm_scale = 1/sqrt(128)`.
Citations: Llama-3-8B `config.json` (`num_attention_heads` 32, `num_key_value_heads` 8,
`hidden_size` 4096, hence head_dim 128, `max_position_embeddings` 8192, `torch_dtype` bfloat16);
Qwen3-8B `config.json` (`num_attention_heads` 32, `num_key_value_heads` 8, `head_dim` 128).
The two var axes are the micro-batch size (1-64) and the sequence length (1-8192, the Llama-3 pre-training
context).

## Inputs

`q`, `k`, `v` and `do` are standard normal bfloat16. `inputs.py` derives `o` and `lse` from `q`, `k`, `v`:
scores in float32, `lse = logsumexp` over the causal prefix, `o = softmax @ v` rounded to bfloat16,
so `o` and `lse` are consistent with `q`, `k`, `v` the way a forward pass stores them. The reference does not use
the saved `o` or `lse`: it recomputes the attention and its exact gradient in float64 from `q`, `k`, `v`
and `do`, so the baseline's use of the bfloat16 `o` in `delta` and of the float32 `lse` is part of
its error, as in training.

## Workloads

Performance cases are training micro-batches of 1 to 64 sequences of 1k to 8k tokens (1k to 131k tokens per
micro-batch, a 128x span). The shortest case, one 1024-token sequence, is 2.1e10 FLOP of causal backward attention
(5 matrix products of 2 x S^2/2 x 128 FLOP per head, 32 heads): 21 us at 1 PFLOP/s plus about 4 us of fixed cost, and the
fused kernel is far from peak there (8 KV heads x 8 key blocks = 64 programs), so it is estimated at
40 us or more. The largest, 64 x 2048 tokens, is 5.5e12 FLOP, about 6 ms at peak and an estimated 15-30 ms.
Sequences shorter than 1024 tokens are correctness-only. The correctness boundary cases are two tokens,
17 and 129 (one past a tile), 127 and 128, 200, 333, 513, 640 and 1000 tokens, with batches of 1-4, and one case with a
non-default `sm_scale = 0.125`.

## Why the sequence axis starts at 2

With one token per sequence every query attends to exactly one key, so `P = 1` and `dS = P * (dP - delta) = 0`
(`delta = do . o = do . v` there): no gradient flows through the scores, `dq` and `dk` are identically zero, and there is nothing to validate. At `seqlen = 2` the causal row 1 attends to two keys, so its `dS` is nonzero
(`dq` of row 1 and `dk` of key 0 are nonzero; only row 0, which sees one key, has `dq = 0`). The axis minimum is therefore 2 and
the boundary case is `c_b1_s2`. Every other correctness case has `seqlen >= 17`.

## Error bounds

**`dk` and `dv`** use `baseline_relative` with `k = 4` (default 2). The baseline's error comes from rounding `P` and
`dS` to bfloat16 before the matrix products and from the bfloat16 `o` in `delta`. In a CPU emulation of the baseline
arithmetic (1024 tokens, 8 query heads over 2 KV heads, head_dim 128, causal, scripted with the framework's own
`compare_tensor`), a more accurate implementation that keeps `P` and `dS` in float32 stayed below the bound at `k = 3`
(peak 0.92x) and `k = 4` (0.72x) on `dq`, `dk` and `dv`, and exceeded the default `k = 2` bound on `dq` (1.32x). At
`k = 4` the emulation rejects a shifted `dv`.

**`dq`** uses `sqnr` with `min_db = 45`. Query row 0 of every sequence attends to exactly one key (itself), so
`P = 1`, `dS = P * (dP - delta) = 0` and the true `dq` row is exactly 0. The baseline computes `dS` as the difference
of two float32 sums that cancel (`dP = do . v` on the matrix unit, `delta = do . o` with `o = v` exactly), so its row is
rounding noise of order 1e-8 instead of 0, and the noise of another correct float32 implementation is independent
of it. Under `baseline_relative` the limit of such a row is `k` times the baseline's own noise, so the ratio of two independent
scalar noises (one per head, each multiplying the row of `k`) exceeds it for about 16% of the heads at `k = 4`:
a float32 PyTorch evaluation of the reference violated the `dq` bound in the shortest cases and in some 2048-token
cases (at `seqlen = 2`, 1.1% of the `dq` elements, within the 50% that row 0 makes up), while a row with two or
more keys carries the baseline's bfloat16 `o` error (about 1e-3 relative), which dominates float32 noise. Raising `k` or the violation
fraction does not remove the heavy tail (and at `seqlen = 2` row 0 is half of `dq`, above the 1% cap on
`max_violation_fraction`). The SQNR of the whole output measures what the bound exists to measure, the error of the
gradient, and ignores noise-level rows. In the CPU emulation (64, 333 and 1024 tokens) the baseline arithmetic
reaches 51 dB on `dq`, and other correct variants (float32 `P` and `dS`, only `dS` rounded, only `P` rounded) 51 to 58 dB.
Controls on `dq` stay below 45 dB: all zeros 0 dB, a missing half 5-8 dB, a 16-ulp shift of every element 20.5 dB,
FP8 (e4m3) rounding 31.5 dB, a 1% scale error 40 dB, a 0.3% scale error 47.5 dB; a poisoned or unwritten output is non-finite. A scale error
below about 0.5% is accepted for `dq` alone (`dk` and `dv`, which depend on the same `dS`, are bounded per element,
and a wrong `sm_scale` is rejected there). Rows missing from `dq` in the short correctness cases (at most 640 rows per
sequence) lower the SQNR far below 45 dB, so tile-edge errors are caught by those cases rather than by the large ones.
