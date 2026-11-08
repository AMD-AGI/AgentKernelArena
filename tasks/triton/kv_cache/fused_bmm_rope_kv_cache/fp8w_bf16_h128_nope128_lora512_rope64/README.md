# Fused FP8 BMM + RoPE + MLA cache write, 128 heads, nope 128, latent 512, rope 64

The front end of a multi-head latent attention (MLA) decode step with the weight-absorbed key projection
in FP8. One launch does two independent jobs:

1. **Projection.** For every query head `h`, `q_nope[h]` (`[tokens, 128]` bfloat16) is quantized inside the
   operator to `float8_e4m3fn` with one float32 scale per token and head (`scale = max(max|x|, 1e-10) * (1 / 448)`,
   codes `fp8(clamp(x * (1 / scale), -448, 448))`), multiplied with the fp8 weight `w_k[h]` (`[512, 128]`),
   accumulated in float32, scaled by the token scale and the per-tensor weight scale `w_k_scale`, and rounded
   to bfloat16 into `q_out[:, h, :512]`.
2. **RoPE and cache.** Interleaved (GPT-J style) rotary embedding of `q_pe` (128 heads) fills
   `q_out[:, :, 512:]`; `k_pe` of the single latent KV head is rotated into `k_pe_out`; `[k_nope | rope(k_pe)]`
   is written to `kv_cache[slot_mapping[t]]`. cos/sin rows are gathered by `positions[t]`.

`kv_cache` is `mutable` and compared in full, so slots that no token addresses (about a fifth of the cache: it holds `n_tokens + n_tokens // 4 + 16` slots) must keep their zero initial value.

## Baseline

AITER's Triton `_fused_fp8_bmm_rope_cat_and_cache_mla_kernel` (MIT, `baseline/LICENSE`), the fusion of
`batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant` with `fused_qk_rope_cat_and_cache_mla`.
The kernel, `_unit_rope`, the 1D rotation helpers and the N=512, K=128 gfx950 tuning table are copied (the table
inlined as a function of M in the lookup order of `get_gemm_config`); the launch code is reduced to `run()`.
Not vendored: the FP4 variant and its split-K reduce kernel, and the decode-copy (`decode_q_pe_out`) and zeros
outputs, which `num_decode_toks_for_zeros = 0` disables, as is the cache scale (`k_scale=None`, bfloat16 cache).
Pointers of the disabled outputs are passed as `q_out` (the matching branches are constexpr-off or never run).

Provenance notes: the kernel file carries the AITER SPDX header and `Copyright (C) 2026, Advanced Micro Devices,
Inc.`, kept verbatim; the rotation helpers come from `rope.py` (`Copyright (C) 2024-2026`, same header, kept in
the vendored file's docstring); `make_kernel_repr` comes from `utils/_triton/kernel_repr.py`, which has no
header upstream, so the repository MIT license applies. No third-party origin note was found in these files
or in their history (`git log --follow` ends at AITER commits "Fused rope_kv and bmm" and "Triton RoPE Fwd
Kernels"). The kernel's only callers in the pinned checkouts are AITER's own tests (which use other, smaller
dimensions), so the geometry below is DeepSeek-V3's: the AITER N=512, K=128 gfx950 table of this operator
matches the absorbed q_nope projection of that model.

## Scenario and config citations

DeepSeek-V3 `config.json` (Hugging Face): `num_attention_heads` 128, `qk_nope_head_dim` 128,
`qk_rope_head_dim` 64, `kv_lora_rank` 512, `rope_theta` 10000. The MLA latent cache has one KV head of width
`kv_lora_rank + qk_rope_head_dim` = 576. `w_k` is the key half of `kv_b_proj` absorbed into the query
(`[128 heads, 512, 128]`), stored as fp8 with a single dynamic per-tensor scale (amax over all heads), the way
AITER's MLA path quantizes it. The rope tables hold `pos * 10000^(-2i/64)` rows for 32768 positions, in
bfloat16, without the YaRN scaling of the full model (the tables are task inputs, so YaRN changes only their values, not the kernel's work; position handling is unchanged by it, only the table
values); `positions` are drawn uniformly over the table.

## Workloads

Token counts of decode batches (64 to 2048 with speculative tokens) and prefill chunks up to 6400 tokens: eight
performance cases (64, 128, 256, 512, 1024, 2048, 4096, 6400), a 100x span. Correctness-only: 1, 3, 17 (boundary
cases), 100 (unaligned decode batch), 1000 and 3000 (unaligned chunks, the latter between tuned rows). The cache holds
`n_tokens + n_tokens // 4 + 16` slots, and the tokens occupy a random subset without repeats.

**Timing.** Per token the operator moves about 195 KB (q_nope 32 KB, q_pe 16 KB, q_out 147 KB) plus 8.4 MB of
weights per launch, but the real cost has a large fixed part: measured baseline medians on MI355X (cold-cache CUDA
graph) were 80.5 us at 192 tokens, 104 us at 256, 126 us at 512, 240 us
at 1024, 466 us at 2048, 927 us at 4096, 1855 us at 8192 and 5746 us at 25600. The roofline (about 10 us at 96 tokens)
underestimates small batches badly, so 64 tokens are estimated at 30 to 60 us, 128 at about 70 us, and 6400 at about
1.45 ms (scaled from 4096 and 8192).

**Memory sizing.** The harness keeps the float64 reference of every checked tensor and about ten float64
comparison temporaries of the same size, in addition to the baseline and candidate outputs. `q_out` dominates:
128 x 576 x 8 B = 590 KB per token per float64 copy (14.06 GiB at 25600 tokens). The measured peak of candidate correctness was at least 249 GB at 25600 tokens, i.e. 9.7 MB per
token. The cache is small (576 x 8 B = 4.6 KB per slot, 1.25 slots per token) and not the dominant term. Peak estimates at
9.7 MB per token: 0.6 GB at 64 tokens, 1.2 at 128, 2.5 at 256, 5.0 at 512, 9.9 at 1024, 19.9 at 2048, 39.7 at 4096
and 62 GB at 6400 (the largest case).

## Comparison

Default `baseline_relative` for `q_out`, `k_pe_out` and `kv_cache`. The reference models the in-operator
quantization (float32 scale and reciprocal, round-to-nearest-even e4m3 codes, exact products) so the baseline's
error against it is only accumulation order and bfloat16 output rounding; an implementation that skips the
activation quantization differs from the reference by the fp8 quantization error (about 3% of each element)
and is rejected, which is intended: the quantization is part of the operator. Error bounds are per row of 576
elements, so the exactly copied latent part of a cache row shares the row's budget with the rotated part. The
baseline computes RoPE in bfloat16 arithmetic (cos/sin are bfloat16), so a candidate that computes it in float32 is
more accurate than the baseline and passes. Negative slots are excluded (the baseline leaves `k_pe_out` unwritten
for them).
