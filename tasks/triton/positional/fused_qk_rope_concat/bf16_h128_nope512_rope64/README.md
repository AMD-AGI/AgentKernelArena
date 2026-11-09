# Fused QK RoPE + concat, bfloat16, 128 heads, latent 512, rope 64

The query/key assembly of a multi-head latent attention (MLA) decode step. Per token it applies interleaved
(GPT-J style) rotary embedding to `q_pe` (128 heads) and `k_pe` (one latent KV head), with cos/sin rows
gathered by `positions`, and concatenates: `q_out = [q_nope | rope(q_pe)]` (`[tokens, 128, 576]`) and
`k_out = [k_nope | rope(k_pe)]` (`[tokens, 1, 576]`). `q_nope` and `k_nope` are the latent 512-wide
vectors of the weight-absorbed decode path. There is no cache write here: the task sits next to
`kv_cache/fused_qk_rope_cache_mla` (same math plus the cache store) and covers the variant that only returns
the concatenated tensors.

## Baseline

AITER's Triton `_qk_rope_cat_kernel` (`fused_qk_rope_cat`), one program per (token, query head); the program
of every `QH_PER_KH`-th head also handles the shared key head. The plain-concat variant (`fused_qk_cat`) is a
copy and is not part of this task; the "rope + cat" path is the one a model uses with positions.

Provenance: `fused_qk_concat.py` has no SPDX or copyright header upstream, so the repository MIT license
(`baseline/LICENSE`) applies to it. The two rotation helpers come from `rope/rope.py` (`SPDX-License-Identifier:
MIT`, `Copyright (C) 2024-2026, Advanced Micro Devices, Inc.`, reproduced in the vendored file's docstring). No
third-party origin note was found in either file; `git log --follow` of `fused_qk_concat.py` ends at the AITER
commit "[Triton] DS fused custom ops". In the pinned SGLang only `rocm_linear_utils.py` re-exports
`fused_qk_rope_cat`; no call site exists in the pinned SGLang or vLLM, so the call geometry is AITER's own test
(`op_tests/triton_tests/fusions/test_fused_qk_concat.py`: nope 512, rope width 64, bfloat16, half-width cos/sin, both
rotation styles) with DeepSeek-V3's head count.

## Scenario and config citations

DeepSeek-V3 `config.json` (Hugging Face): `num_attention_heads` 128, `kv_lora_rank` 512, `qk_rope_head_dim` 64,
`rope_theta` 10000. The MLA latent cache has one KV head. The tables hold `pos * 10000^(-2i/64)` rows for 32768
positions in bfloat16 (no YaRN scaling of the full model); `positions` are drawn uniformly over the table.

## Workloads

Decode batches (128 to 2048 requests with speculative tokens) and prefill chunks up to 16384 tokens, a 128x span.
Per token the operator moves about 296 KB (q_nope 131 KB and q_pe 16 KB read; q_out 147 KB written; the key
side adds about 2.5 KB), so 128 tokens take about 12 us (38 MB / 4.5e12 B/s + 4 us) and 16384 tokens about 1.1 ms.
Smaller batches are correctness-only (1, 3, 17 and 100 are the boundary cases; 1000 is an unaligned chunk).

## Comparison

Default `baseline_relative` on both outputs. Error bounds are per row of 576 elements, so the exactly copied
latent part shares each row's budget with the rotated part. The baseline computes RoPE in bfloat16 arithmetic
(cos/sin are bfloat16), so a candidate computing in float32 is more accurate than the baseline and passes.
