# Paged decode attention, BF16, 16 query heads / 1 KV head, head size 128, 16-token pages (Gluon baseline)

Single-token (decode) grouped-query attention over vLLM's paged KV cache. For each sequence `s`
and query head `h`,

```text
out[s, h] = softmax(scale * q[s, h] . K_s^T) . V_s
```

where `K_s`, `V_s` are the first `seq_lens[s]` cached tokens of sequence `s`, all 16 query heads
share the single KV head, `scale = 128 ** -0.5`, and the result is rounded to bfloat16. The
interface, layouts, reference and input generator are those of the sibling task
`tasks/hip/attention/paged_attention_decode/bf16_q16_kv1_d128_p16` (same operator, same scenario), so
every baseline language is scored on the same semantics. Only the axis ranges and the workloads differ.

## Production origin

The baseline is AITER's Gluon paged-attention decode, `aiter/ops/triton/gluon/pa_decode_gluon.py` from
ROCm/aiter at 652f2d4b768239aefd515c929e84817681ff598d (MIT, `baseline/LICENSE`; the file has the AMD
SPDX/copyright header, which is kept verbatim, and no third-party origin note). Sixteen query heads
sharing one KV head with head size 128 is the per-rank geometry of Qwen3-235B-A22B (`config.json`:
`num_attention_heads` 64, `num_key_value_heads` 4, `head_dim` 128) at tensor parallel 4 and of
Llama-3.1-405B (`num_attention_heads` 128, `num_key_value_heads` 8, `head_dim` 128) at tensor
parallel 8. AITER's op test (`op_tests/triton_tests/test_pa_decode_gluon.py`) covers head size 128,
16-token pages and the (16 query, 1 KV) head configuration.

## Interface and layouts

See the sibling task for the full description. `query[num_seqs, 16, 128]`,
`key_cache[num_blocks, 1, 16, 16, 8]` (element `d` of the token in `(page, slot)` at
`[page, kv_head, d // 8, slot, d % 8]`), `value_cache[num_blocks, 1, 128, 16]` (at
`[page, kv_head, d, slot]`, the non-transposed value layout of `pa_decode_gluon`),
`block_table[num_seqs, ceil(max_seq_len / 16)]` and `seq_lens[num_seqs]`. Context lengths are uniform in
`[1, max_seq_len]` with one sequence at `max_seq_len`. Page 0 is the reserved null page; every slot that
holds no token (page tails, unused pages, page 0) holds finite stale values 64 times larger than the
token values and must not reach the result.

## Baseline

`baseline/kernel.py` vendors the persistent-scheduling decode kernel for one KV head,
`paged_attention_decode_sliding_window_head_1`, and its helpers (`define_layout`,
`store_temporary_result`, the scheduling-hint imports) unchanged. This is the kernel `pa_decode_gluon`
launches by default (`ps=True`) when `num_kv_heads == 1`.

`pa_decode_gluon` splits each sequence across `max_context_partition_num` programs and then merges the
partial results with a reduce kernel (`paged_attention_decode_v2_reduce_kernel` in Triton, or a FlyDSL
or C++ reduce). A Gluon baseline may launch only Gluon kernels, so this task uses the wrapper's
one-shot mode, `max_context_partition_num = 1`: one program per sequence loops over the whole context
in 256-token partitions and writes the normalized output directly, with no reduce launch.
The task keeps that mode honest to production by choosing the batch sizes where it applies: AITER's own
split-count heuristic (`get_recommended_splits`, `ceil(2 * CUs / (num_seqs * num_kv_heads))` capped at 8)
returns 1 from 512 sequences on a 256-CU MI355X, and returns 2 to 8 below that, where production adds the
reduce stage. Small batches would therefore need a Gluon port of the reduce; they are covered as
correctness-only cases, not timed. `run()` replaces `pa_decode_gluon` and its kernel-selection wrapper with
the same dispatch for this configuration (bf16 query and cache, no quantization scales, no sinks, no
sliding window, query length 1, `KV_BLOCK_SIZE` 16, `CONTEXT_PARTITION_SIZE` 256, `CDNA_VERSION` 4); the
`exp_sums`/`max_logits` buffers the wrapper allocates are kept (small, unused in one-shot mode), and the
temporary-output buffer, which only the split path uses, is not allocated. The aiter, flydsl and csrc
imports, the other kernels and the fallback path that printed a warning when Gluon is unavailable are
removed; the arch query is inlined.

## Workloads

Ten timed cases, 512 or 1024 sequences with `max_seq_len` 256 to 16384; they span 128x in
`num_seqs * max_seq_len` (131072 to 16.8M). The largest, 1024 sequences of up to 16k tokens, holds about
8.4M cached tokens (4.3 GB of keys and 4.3 GB of values allocated per layer), the same order as one MI355X
rank's KV capacity for Qwen3-235B-A22B at FP8 weights (about 4M tokens summed over its layers, so the
per-layer cache of this case is a benchmark upper bound, not a serving configuration). Estimated baseline time:
about 15 us for `s512_l256` (34 MB of cache read plus launch; the smallest timed case) and about 1 ms for `s1024_l16384` (4.3 GB read
at near-peak bandwidth). Every smaller or low-concurrency shape runs below the timing floor or needs the
split path and is correctness-only: single requests, the page and partition tails (17 and 257 tokens),
contexts 255 and 1025, unaligned batches and contexts (37 x 1000, 7 x 9999), and one request at 16384
tokens, which is 64 sequential partitions in one program.

## Reference and bound

The float64 reference of the sibling task is reused. The default `baseline_relative` bound applies;
the baseline rounds the probabilities to bfloat16 before the PV product, which is part of its own error.

## Launch

A one-program-per-sequence launch
makes the long-context cases latency-bound on a single sequence's loop, so `s1_l16384` is slow but
correct. The `triton.runtime.driver` arch query in `get_cdna_version` runs on the first call.
