# Gather + kv_b projection of the MLA latent prefix, bfloat16 cache, fp8 block-scaled weight, 16 heads

The prefix expansion of chunked prefill in a multi-head latent attention (MLA) model. A request's cached
prefix is stored as one 576-wide latent row per token (512 latent `kv_c` + 64 rotated `k_pe`, bfloat16) in a paged
cache with block size 1. To run ordinary multi-head attention over the prefix, the kernel gathers the
rows of every request (`kv_indices[kv_indptr[b]:kv_indptr[b + 1]]`), multiplies `kv_c` with the `kv_b_proj`
weight and writes keys and values per head:

```text
proj[t, h, :]      = kv_c[t] @ W[256 h : 256 h + 256]^T         # W dequantized: fp8 code * scale of its 128x128 block
k_prefix[t, h, :]  = [proj[t, h, :128] | k_pe[t]]               # k_pe is shared by the heads
v_prefix[t, h, :]  = proj[t, h, 128:]
```

Tokens are packed request after request; `kv_prefix_sum_context_lens` holds each request's first output row
(with block size 1 it equals `kv_indptr`, both are passed as production does).

## Baseline

AITER's Triton `_triton_gather_kv_b_proj` (MIT, `baseline/LICENSE`): one program per (request, head) that holds
that head's 128x512 key and value weights in registers (converted to bfloat16), then walks the request in chunks
of 16 tokens with eight 16x128x128 `tl.dot`s per chunk and applies the block scales per 128-wide K segment. The
preshuffled-weight path (`WEIGHT_PRESHUFFLE=True`, AITER's test default) is used; the weight is therefore delivered in
the `(16, 16)` MFMA layout by the task's `layout` transform (`to_kernel_layout`, the one-byte case of
`aiter.ops.shuffle.shuffle_weight`), outside timing. A CPU emulation of the kernel's in-register unshuffle
(`_load_unshuffle_segment`) applied to this layout reproduces the reference to 2.4e-7 on the 128-wide key and value
parts of a head. The kernel file is copied unchanged (it also holds FP4 and shuffled-cache paths
this scenario does not take), with the SPDX header and `Copyright (C) 2025-2026, Advanced Micro Devices, Inc.`
kept verbatim; the launch code of `gather_kv_b_proj` is reduced to `run()`. No third-party origin note was
found in the file; `git log --follow` shows only AITER commits ("Add gather_kv_b_proj triton kernel", reverted
and re-added as "Feat gen kv prefix preshuffle1"). The pinned SGLang and vLLM have no call site for it; only
AITER's tests (`op_tests/triton_tests/test_gather_kv_b_proj.py`) call it, so the scenario follows that test's
DeepSeek-style geometry (latent 512, rope 64, nope 128, value 128, fp8 weight with `[N/128, 4]` scales, page
size 1 and 16-32).

## Scenario and config citations

DeepSeek-V3 `config.json` (Hugging Face): `num_attention_heads` 128 (16 on each rank of an 8-way tensor
parallel deployment), `qk_nope_head_dim` 128, `v_head_dim` 128, `qk_rope_head_dim` 64, `kv_lora_rank` 512, and its
`quantization_config`: `fp8` (e4m3) with `weight_block_size` [128, 128]. So `kv_b_proj` is
`[16 * (128 + 128), 512]` fp8 with a `[32, 4]` float32 scale. The cache is bfloat16 (the kernel's fp8-cache path with
`k_scale` is another scenario).

## Workloads and inputs

The var axis is the total number of cached prefix tokens of the batch; 16 requests share it with unequal
lengths (every request has at least one token; shares drawn uniformly from 0.5 to 1.5 of the mean), which exercises
requests longer or shorter than a 16-token chunk and a ragged last chunk. Cache rows of all tokens are distinct
random blocks of a 262144-block pool, so the gather is scattered. `kv_c` is normal (std 1, as after the latent
RMSNorm), the weight is N(0, 0.03^2) quantized per 128x128 block with scale = block amax / 448, as the checkpoint
stores it.

Performance cases run from 2048 to 204800 tokens (100x): the expansion writes 10 KB per token (6 KB keys, 4 KB
values) and does 4.2 MFLOP per token; every program also loads 256 KB of weights into registers, so the
smallest case (2048 tokens, about 128 per request) is estimated at 15 to 30 us and 204800 tokens at 1 to 2.5 ms
(the longest request, up to 1.5x the mean, runs its chunks serially). Smaller batches are correctness-only (16 is
the axis minimum, 17 and 1000 and 3001 are non-power-of-two and unaligned; 100 is a short-prefix batch).
Memory at the largest case: about 2 GB of outputs, 8.4 GB for the float64 reference.

## Comparison

Default `baseline_relative` on both outputs. The reference computes the projection in float64 with exactly
dequantized weights; the baseline's error against it is only the bfloat16 rounding of the float32 accumulation,
so a candidate must accumulate at least as accurately. The `k_pe` half of every key row is a copy (exact in the
baseline); error bounds are per row of 192 elements, so a copied row element shares the row's budget.
