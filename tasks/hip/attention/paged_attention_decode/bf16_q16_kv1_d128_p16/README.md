# Paged decode attention, BF16, 16 query heads / 1 KV head, head size 128, 16-token pages

Single-token (decode) grouped-query attention over vLLM's paged KV cache. For each sequence `s`
and query head `h`,

```text
out[s, h] = softmax(scale * q[s, h] . K_s^T) . V_s
```

where `K_s`, `V_s` are the first `seq_lens[s]` cached tokens of sequence `s`, all 16 query heads
share the single KV head, `scale = 128 ** -0.5`, and the result is rounded to bfloat16.

## Production origin

vLLM's ROCm attention backend (`rocm_attn.py`) runs decode through
`chunked_prefill_paged_decode`, which calls the hand-written ROCm paged-attention kernels
(`paged_attention_rocm`) whenever `use_rocm_custom_paged_attention` holds on CDNA GPUs: BF16 or
FP16 queries, head size 64 or 128, 16- or 32-token pages, a query group of 1 to 16 and contexts of
at most 128k tokens. A group of 16 dispatches `paged_attention_ll4mi_QKV_mfma16_kernel`
(per-partition attention over 256-token partitions), followed by
`paged_attention_ll4mi_reduce_kernel` (cross-partition softmax reduction).

Sixteen query heads sharing one KV head per tensor-parallel rank is the per-rank geometry of
Qwen3-235B-A22B (64 query / 4 KV heads) at TP=4 and of Llama-3.1-405B (128 / 8) at TP=8, both with
head size 128. The original arena task captured vLLM decode calls of this geometry with a BF16
cache, 1 to 64 sequences and 1024 to 4096 tokens of context. It also held calls with 32 query /
4 KV heads (group 8, the MFMA16 kernel instantiated for 8) and one 16 / 4 page-tail case (group 4,
the MFMA4 kernel). Head counts are constants of a scenario, so those geometries are not part of
this task: group 4 is the scenario `bf16_q32_kv8_d128_p16`, and group 8 with this kernel would be
a further scenario.

## Interface and layouts

- `query[num_seqs, 16, 128]`: one decode token per sequence.
- `key_cache[num_blocks, 1, 16, 16, 8]`: element `d` of the token in `(page, slot)` is at
  `[page, kv_head, d // 8, slot, d % 8]` (vLLM's ROCm cache layout with 16-byte key vectors).
- `value_cache[num_blocks, 1, 128, 16]`: element `d` of the token in `(page, slot)` is at
  `[page, kv_head, d, slot]`.
- `block_table[num_seqs, ceil(max_seq_len / 16)]`: token `t` of sequence `s` is in page
  `block_table[s, t // 16]`, slot `t % 16`. Pages of a sequence are scattered at random; entries
  past a sequence's last page are 0.
- `seq_lens[num_seqs]`: context lengths. One sequence has `max_seq_len` tokens; the others are
  uniform in `[1, max_seq_len]`, a steady-state batch of long generations from short prompts (the
  regime of reasoning models such as Qwen3-235B).

Page 0 is vLLM's reserved null page. Every cache slot that holds no token -- the tail of a
sequence's last page, pages past its last page and page 0 -- holds stale finite values 64 times
larger than token values, as reused cache pages do in production. The production kernels read the
whole last page and give its tail slots zero probability; an implementation must likewise keep
those slots out of the result. (They are finite rather than NaN because the production kernel
multiplies masked values by a zero probability, which is exact only for finite values.)

## Workloads

Every timed case streams at least about 33 MB of cache, enough to take the baseline over the
10 us timing floor on MI355X: single long-context requests (64k tokens and the kernel's 128k limit),
four requests at 32k tokens, and batches of 16 to 256 sequences. The largest case, 224 sequences
at Qwen3's 32k native context, holds about 3.7M cached tokens, which fits the KV capacity of one
MI355X rank of Qwen3-235B-A22B (FP8 weights, TP=4, about 48 KB of cache per token across its 94
layers). `s256_l1024` and `s128_l8192` reuse the batch sizes and contexts of the original AITER
paged-attention decode task (batch 256 at 1k tokens, batch 128 at 8k tokens), which timed query
groups of 8. The original task's shapes -- 1 to 64 sequences at 1024 or 1025 tokens, the page and
partition tails at 17 and 257 tokens and two sequences at 4096 tokens -- finish in a few
microseconds and are correctness-only, together with contexts that need two or three
cross-partition reduction loops (more than 64 partitions).

The 10 us floor and one rank's KV capacity bound this scenario's timed range from both sides: a
single KV head streams only 512 bytes of cache per token, so the smallest timed case needs about
64k tokens, and the largest batch is near what the cache can hold. The timed cases span 112x in
`num_seqs * max_seq_len`.

## Baseline

`baseline/kernel.hip` vendors vLLM's `csrc/rocm/attention.cu` at commit
`edbc4969a76b25c531f7b9bb16b79984e65ef023`: the gfx9 kernels with their BF16 code path unchanged,
their host-pass stubs, launch macros and `paged_attention_custom_launcher`. The gfx11/gfx12
kernels, the Navi launcher and the FP8 KV-cache, FP8-MFMA and FP8-output branches were removed
(this scenario has a BF16 cache and output). The launcher's GQA switch keeps the production branch
for this scenario (group 16, MFMA16 kernel) and launches on the current HIP stream. `run()`
replaces `paged_attention()`: it allocates `exp_sums`, `max_logits` and `tmp_out` per call as
`chunked_prefill_paged_decode` does, passes no `query_start_loc` (every sequence has one query
token, so the kernels index queries by sequence), and takes `max_seq_len` as the block table's
width in tokens. Because the table holds `ceil(max_seq_len / 16)` pages, that width yields the same
`ceil(max_seq_len / 256)` partitions as the production call. The kernels parallelize over
(sequence, partition, KV head) with 256 threads per partition, so a single KV head gives few
workgroups for short contexts or small batches. Making better use of that small grid is where a
candidate can win.

The comparison uses the default `baseline_relative` bound.
