# Sparse MLA attention, DeepSeek-V3.2 (16 heads, 512 latent + 64 rope, top-k 2048)

Each query token attends to a per-token top-k subset of a paged latent KV cache chosen by the
lightning indexer. The 576-wide cache row is the key in full (latent plus rotary part) and its first
512 entries are the value; all query heads share one latent head. Entries equal to `-1` are ignored.

**Production origin.** AITER's `unified_attention_sparse_mla`, the ROCm MLA path for the DeepSeek-V3.2
sparse attention backends of vLLM and SGLang. The scenario uses the model dimensions: top-k 2048,
`kv_lora_rank` 512, `rope_rank` 64, cache block size 64 and 16 query heads (128 heads at tensor
parallel 8). The previous task used toy sizes (64 to 128 selected rows); these are production sizes.

**Scenario.** One causal prefill sequence of `n_tokens` tokens whose own latent rows fill the cache
(`num_blocks = ceil(n_tokens / 64)`). Token `t` selects `min(t + 1, 2048)` distinct positions from
`0..t`; shorter rows are padded with `-1` (the first 2048 tokens), so masking is always exercised.
`cu_seqlens_q` is `[0, n_tokens]`. Decode-style batches of single-token sequences are a different
scenario and are not covered.

**Workloads.** Performance cases are prefill chunk sizes 128 to 16384; the boundary cases and a
2500-token chunk (partial and full rows together) are correctness-only. The kernel runs one program per
token and iterates all 32 top-k tiles, so even the 128-token case is expected well above the 10 us
floor (to be confirmed by the GPU audit).

**Baseline.** AITER's Triton kernel, unchanged, with AITER's launch code in `run()`. The kernel never
reads the block table or per-sequence lengths it is given, so the task omits those inputs and `run()`
hands the kernel `cu_seqlens_q` in their place.

**Comparison.** Default `baseline_relative`. The reference returns float32 only to bound its memory; it
is computed in float64 internally.
