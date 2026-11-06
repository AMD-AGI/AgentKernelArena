# Fused QK RoPE and MLA cache write, bfloat16, 128 heads, latent width 512, rope width 64

The multi-head latent attention (MLA) front end of a DeepSeek-V3-style decode step. Per token it
applies interleaved (GPT-J style) rotary embedding to `q_pe` (128 heads) and `k_pe` (one latent KV
head) with cos/sin rows gathered by `positions`, writes `[q_nope | rope(q_pe)]` to `q_out`, the
rotated `k_pe` to `k_pe_out`, and `[k_nope | rope(k_pe)]` into the paged MLA cache at `slot_mapping`.
Slots are unique and non-negative (the baseline leaves `k_pe_out` unwritten for a negative-slot token,
so padding is not part of this scenario); the cache is `mutable` and compared in full, so untouched
slots (half of the 32768) must keep their (zero) initial value.

Baseline: AITER's Triton `_fused_qk_rope_cat_and_cache_mla_kernel` (MIT, `baseline/LICENSE`),
vendored with its RoPE and cache-store helpers; FP4 cache branches removed and the launch wrapper
reduced to `run()` for this scenario. In this kernel version `q_nope` and `k_nope` share the latent
width, which is why `q_nope` is 512 wide, as in the weight-absorbed MLA decode path.

Workloads: token counts of decode batches (128 to 2048, including speculative-decode batches) and
prefill chunks up to 16384 tokens at 128 heads. Smaller batches run under the 10 us timing floor
(about 0.3 MB of traffic per token) and are correctness-only. cos/sin are bfloat16 tables of
`pos * 10000^(-2i/64)`, as in the model's rotary cache; the reference rounds nothing but those tables.

Compare: default `baseline_relative` on all three written tensors. Error bounds are per row of 576
elements, so the exactly copied latent part shares the row's RMS error budget with the rotated part.
The baseline computes RoPE in bfloat16 arithmetic (cos/sin are bfloat16), so a candidate that
computes in float32 is more accurate than the baseline and passes.
