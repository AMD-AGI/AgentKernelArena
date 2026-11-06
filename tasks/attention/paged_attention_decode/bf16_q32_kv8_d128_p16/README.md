# Paged decode attention, BF16, 32 query heads / 8 KV heads, head size 128, 16-token pages

Single-token (decode) grouped-query attention over vLLM's paged KV cache. For each sequence `s`
and query head `h`,

```text
out[s, h] = softmax(scale * q[s, h] . K_s^T) . V_s
```

where `K_s`, `V_s` are the first `seq_lens[s]` cached tokens of sequence `s` for KV head `h // 4`,
`scale = 128 ** -0.5`, and the result is rounded to bfloat16.

## Production origin

The scenario comes from a Llama-3.1-8B-Instruct serving session on MI355X (vLLM 0.24.0, TP=1,
concurrency 64, 1024-token prompts and outputs, `max_model_len` 6144, BF16 KV cache), where the
ROCm custom paged-attention decode pair was the largest decode leaf (26.85% of GPU time). The
call chain is `rocm_attn.py` -> `chunked_prefill_paged_decode` -> `paged_attention_rocm`; with a
query group of 4 the launcher picks `paged_attention_ll4mi_QKV_mfma4_kernel` ("mfma4 kernel is
faster than mfma16 for gqa_ratio <= 4"), and the trace records exactly
`paged_attention_ll4mi_QKV_mfma4_kernel<bf16, bf16, kAuto, bf16, 16, 128, 256, false, 4>` followed
by `paged_attention_ll4mi_reduce_kernel`. Qwen3-8B has the same geometry and workload.

The original arena task measured AITER's MFMA16 copy of this kernel family because its runtime
image shipped no vLLM C++ source. This task vendors the vLLM kernels the session actually ran.
The original task also timed the same sessions' other geometries, Qwen3-0.6B (16 / 8 heads,
group 2, MFMA4) and Qwen3-14B-FP8 (40 / 8 heads, group 5, MFMA16). Head counts are constants of a
scenario, so those geometries are not part of this task; each would be a further scenario.

## Interface and layouts

- `query[num_seqs, 32, 128]`: one decode token per sequence.
- `key_cache[num_blocks, 8, 16, 16, 8]`: element `d` of the token in `(page, slot)` for KV head
  `k` is at `[page, k, d // 8, slot, d % 8]` (vLLM's ROCm cache layout with 16-byte key vectors).
- `value_cache[num_blocks, 8, 128, 16]`: element `d` of the token in `(page, slot)` for KV head
  `k` is at `[page, k, d, slot]`.
- `block_table[num_seqs, ceil(max_seq_len / 16)]`: token `t` of sequence `s` is in page
  `block_table[s, t // 16]`, slot `t % 16`. Pages of a sequence are scattered at random; entries
  past a sequence's last page are 0.
- `seq_lens[num_seqs]`: context lengths. One sequence has `max_seq_len` tokens; the others are
  between half of it and all of it. That is the session's steady state: 1024-token prompts with up
  to 1024 generated tokens give contexts of 1024 to 2048 (`s64_l2048`).

Page 0 is vLLM's reserved null page. Every cache slot that holds no token -- the tail of a
sequence's last page, pages past its last page and page 0 -- holds stale finite values 64 times
larger than token values, as reused cache pages do in production. The production kernels read the
whole last page and give its tail slots zero probability; an implementation must likewise keep
those slots out of the result. (They are finite rather than NaN because the production kernel
multiplies masked values by a zero probability, which is exact only for finite values.)

## Workloads

The session trace records the query and cache shapes but not the context lengths of its
graph-captured decode launches; the original task reconstructed them from the serving regime as
64 sequences at 1024, 1536 and 2048 tokens (every sequence at that length), kept here as
`s64_l1024`, `s64_l1536` and `s64_l2048` with that length as the longest context of the batch and
the other contexts between half of it and all of it. The other timed cases cover vLLM's decode batch range up to its default limit of 256
sequences, contexts up to the session's `max_model_len` of 6144 and one 32k-token request. The
largest, 256 sequences at 6144 tokens, holds about 1.2M cached tokens (128 KB each across the
model's 32 layers), within one MI355X's KV capacity for this model. Every timed case streams at
least about 44 MB of cache, above the 10 us timing floor. Single short requests and tiny batches
are correctness-only, as are a 16385-token context (the first that needs two cross-partition
reduction loops) and a 131072-token context (Llama-3.1's limit and the kernel's eight-loop
maximum).

## Baseline

`baseline/kernel.hip` vendors vLLM's `csrc/rocm/attention.cu` at commit
`edbc4969a76b25c531f7b9bb16b79984e65ef023`: the gfx9 kernels with their BF16 code path unchanged,
their host-pass stubs, launch macros and `paged_attention_custom_launcher`. The gfx11/gfx12
kernels, the Navi launcher and the FP8 KV-cache, FP8-MFMA and FP8-output branches were removed
(this scenario has a BF16 cache and output). The launcher's GQA switch keeps the production branch
for this scenario (group 4, MFMA4 kernel) and launches on the current HIP stream. `run()` replaces
`paged_attention()`: it allocates `exp_sums`, `max_logits` and `tmp_out` per call as
`chunked_prefill_paged_decode` does, passes no `query_start_loc` (every sequence has one query
token, so the kernels index queries by sequence), and takes `max_seq_len` as the block table's
width in tokens. Because the table holds `ceil(max_seq_len / 16)` pages, that width yields the same
`ceil(max_seq_len / 256)` partitions as the production call. The MFMA4 kernel gives each
256-token partition of each (sequence, KV head) one workgroup that computes all four query heads
with 4x4x4 MFMAs, and the reduction kernel then merges the partitions.

The comparison uses the default `baseline_relative` bound.
