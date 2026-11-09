# Fused QKV split + QK RMSNorm + partial RoPE + paged KV write, 16 query / 2 KV heads, head size 256, gated

The attention input preparation of a Qwen3-Next full-attention layer in one launch. The QKV GEMM returns
`qkv [tokens, 9216]`: per query head the query and its output gate (interleaved `[q_h0, g_h0, q_h1, g_h1, ...]`,
32 x 256 values), then 2 key heads and 2 value heads. The operator

- RMS-normalizes every query and key head over its 256 values with Gemma-style weights
  (`x * rsqrt(mean(x^2) + eps) * (1 + w)`, as vLLM's `Qwen3NextRMSNorm` is `GemmaRMSNorm`);
- rotates the first 64 values of each q/k head with the NeoX (rotate-half) embedding, the cos/sin row coming from
  `positions[t]`; the remaining 192 values keep their normalized value (partial rotary);
- writes `q`, `gate`, `k`, `v` as `[tokens, heads, 256]`;
- writes `k` and `v` to the paged caches `[8192 blocks, 16, 2, 256]` (NHD) at `slot_mapping[t]` (no cache scale).

`key_cache` and `value_cache` are `mutable` and compared in full, so slots that no token addresses keep their zero
initial value (all 131072 slots are used only by the largest case).

## Baseline

AITER's Triton `_fused_qkv_split_qk_norm_rope_cache_kernel` (MIT, `baseline/LICENSE`) with its partial-rotation
path (`ROTARY_SPAN = 64 < 256`: the rotated partner is reloaded from `qkv` at the swapped position and scaled
with the same inverse RMS). The whole kernel file is copied unchanged; the wrapper is reduced to `run()` and the
tile heuristic `infer_rope_cache_triton_block_t` (it reads the compute-unit count, memoized here with
`functools.lru_cache`; 256 on the MI355X, so BLOCK_T is 1 for T = 1 and 32 from 16384 tokens on). Neither the kernel
nor the wrapper has a header upstream, so the repository MIT license applies; the two rotation helpers
come from `rope.py` (`SPDX-License-Identifier: MIT`, `Copyright (C) 2024-2026, Advanced Micro Devices, Inc.`,
reproduced in the vendored file's docstring). Third-party origin: the wrapper's docstring says the BLOCK_T
heuristic is the same as vLLM's `calc_rows_per_block` (Apache-2.0); this is a one-line formula whose code is not
copied. The kernel's history starts at the AITER commit "feat(triton/rope): fused QKV split, QK RMSNorm, RoPE, and
paged KV cache" (no earlier origin). Not covered by this scenario (other `ENABLE_GATED_Q`, `IS_NEOX`,
`REUSE_FREQS_FRONT_PART`, layout and scale flags of the same kernel): blocked gate layout, GPT-J rotation,
full rotary, HND cache, `k_scale`/`v_scale`.

## Scenario and config citations

Qwen3-Next-80B-A3B `config.json` (Hugging Face): `num_attention_heads` 16, `num_key_value_heads` 2, `head_dim`
256, `partial_rotary_factor` 0.25 (rotary width 64), `rope_theta` 10000000, `rms_norm_eps` 1e-6,
`max_position_embeddings` 262144. The config has no `attn_output_gate` key; the output gate comes from vLLM, whose
`qwen3_next.py` (line 263, commit `edbc4969a76b25c531f7b9bb16b79984e65ef023`) reads
`getattr(config, "attn_output_gate", True)`, so it is on by default. The interleaved gate layout and the `(1 + w)` norm are also those
of vLLM's `qwen3_next.py` (`q_gate.view(..., num_heads, head_dim * 2)` then `chunk(2)`). The cos/sin tables hold
the 32 first-half angles `pos * 10000000^(-2i/64)` for 262144 positions in bfloat16; `positions` are drawn
uniformly over the table. `q_weight` and `k_weight` are zero-centered (normal, std 0.3), as Gemma-style weights are
stored, so `1 + w` stays well away from zero. The cache uses the NHD layout with 16-token blocks (vLLM's default
block size).

## Workloads

Decode batches and prefill chunks from 1024 to 131072 tokens (128x). Per token the operator moves about 38.5 KB
(qkv 18.4 KB, q and gate 16 KB, k, v and the cache copies 4 KB), so 1024 tokens take about 13 us (39 MB / 4.5e12
B/s + 4 us) and 131072 tokens about 1.1 ms. Smaller batches are correctness-only (1, 3, 17 and 100 are the
boundary cases; 1000 is an unaligned chunk). Slots are a random subset of the cache without repeats.

## Comparison

`q`, `k` and `key_cache` use the default `baseline_relative` bound; error is per row of 256 elements. The
baseline rounds the normalized value to bfloat16 before the rotation and multiplies by the bfloat16 cos/sin
in bfloat16, so a candidate that normalizes and rotates in float32 is more accurate and passes. `gate`, `v` and
`value_cache` are pure copies of input values (the baseline's scale factor is the integer 1), so they use `exact`
(QA bound-override warnings, intended): any deviation would be a wrong index, not a rounding difference.
