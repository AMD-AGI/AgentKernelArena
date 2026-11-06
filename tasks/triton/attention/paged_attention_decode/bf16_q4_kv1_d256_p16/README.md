# Paged decode attention, BF16, 4 query heads / 1 KV head, head size 256, 16-token pages

Single-token (decode) grouped-query attention over vLLM's paged KV cache. For each sequence `s`
and query head `h`,

```text
out[s, h] = softmax(scale * q[s, h] . K_s^T) . V_s
```

where `K_s`, `V_s` are the first `seq_lens[s]` cached tokens of sequence `s`, all four query heads
share the single KV head, and the result is rounded to bfloat16.

## Production origin

The scenario comes from a Qwen3.5-122B-A10B-FP8 serving session on MI355X (vLLM 0.24.0, TP=8,
concurrency 64, 1024 input / 1024 output tokens, `max_model_len` 6144), where vLLM's
`kernel_paged_attention_2d` was the single largest GPU kernel (15.8% of the timeline). Per rank
the model has 4 query heads, 1 KV head and head size 256 with a BF16 KV cache. The ROCm attention
backend (`rocm_attn.py`) calls `chunked_prefill_paged_decode`; its hand-written ROCm
paged-attention kernel only supports head sizes 64 and 128 on gfx9
(`use_rocm_custom_paged_attention`), so head size 256 always takes this Triton fallback.

The 16-token page size is vLLM's default block size, taken from the original harness; the session
trace records only the query and KV-step shapes. Qwen3.5 is a hybrid model, and vLLM enlarges the
attention page of a hybrid model until it holds one GatedDeltaNet state. From the published model
configuration at TP=8, with its float32 SSM state and no prefix caching, that is 528 tokens. The
kernel would then walk 32-token tiles over 528-token pages. This task keeps the original
16-token-page scenario of the same kernel.

The model's attention uses no ALiBi, sliding window, sinks or FP8 output, so neither does this
scenario. The older generic test of this kernel used FP16 shapes with head sizes 64-128, ALiBi, a
sliding window, mixed prefill/decode batches filtered by query length, and a strided cache. Those
are dispatch options of other models and are not part of this task.

## Interface and layouts

- `query[num_seqs, 4, 256]`: one decode token per sequence.
- `key_cache[num_blocks, 1, 32, 16, 8]`: element `d` of the token in `(page, slot)` is at
  `[page, kv_head, d // 8, slot, d % 8]` (vLLM's `PagedAttention.split_kv_cache` view of the ROCm
  cache, 16-byte key vectors).
- `value_cache[num_blocks, 1, 256, 16]`: element `d` of the token in `(page, slot)` is at
  `[page, kv_head, d, slot]`.
- `block_table[num_seqs, ceil(max_seq_len / 16)]`: token `t` of sequence `s` is in page
  `block_table[s, t // 16]`, slot `t % 16`. Pages of a sequence are scattered at random;
  entries past a sequence's last page are 0.
- `seq_lens[num_seqs]`: context lengths. The longest sequence has `max_seq_len` tokens; the others
  have between half of it and all of it, as in a steady-state decode batch.

Page 0 is vLLM's reserved null page. Every cache slot that holds no token -- the tail of a
sequence's last page, pages past its last page and page 0 -- is NaN, because unwritten cache
memory may hold garbage in production (the production kernel masks its K/V loads for exactly this
reason). An implementation must not let those slots reach the result.

## Workloads

Batch sizes cover vLLM's decode CUDA-graph range (1 to 256 timed, up to 512 for correctness) and
context lengths reach the session's `max_model_len` of 6144 (one long-context correctness case at
32768). `s64_l1024`, `s64_l2048` and `s64_l3072` are the session's own shapes; the session trace
records the query and KV-step shapes but not the context lengths, which the original harness
reconstructed from the serving regime. Batches with very short contexts finish in a few
microseconds and are correctness-only.

## Baseline

`baseline/kernel.py` vendors `kernel_paged_attention_2d` unchanged from vLLM
`vllm/v1/attention/ops/chunked_prefill_paged_decode.py` at commit
`edbc4969a76b25c531f7b9bb16b79984e65ef023`. `run()` is the Triton branch of
`chunked_prefill_paged_decode` for this scenario with its production dispatch constants
(`TRITON_BLOCK_SIZE = 16`, query group padded to 16, no ALiBi, sliding window, sinks or FP8
output). Because every sequence has exactly one query token, the launch indexes queries by
sequence (`filter_by_query_len=False`) instead of reading an `arange` `query_start_loc`; the kernel
code path is otherwise identical. The kernel launches one program per (sequence, KV head) and walks
the context in 16-token tiles with an online softmax, so it parallelizes poorly at small batch
sizes, which is the optimization opportunity this task measures.

## Comparison

`out` uses `baseline_relative` with `k: 4` instead of the default `k: 2`. The baseline rounds
the probabilities of each 16-token tile to bfloat16, relative to the running maximum, before the PV
product. Its error is therefore about one bfloat16 rounding of the output and varies randomly from
element to element. A candidate that moves this rounding is just as accurate but makes its errors
at different elements. Examples are a different tile size, split-K partials (flash decoding) or a
two-pass softmax.

A CPU emulation of this rounding compared such candidates with the exact reference. It covered
batches of 16-256 sequences with contexts of 1024-16384 tokens, using 16- to 128-token tiles and
splits of 128-1024 tokens. With `k: 2`, most of these candidates failed: 1-25 elements were out of
bounds, and the largest error reached 1.40 times the bound. With `k: 3` the largest error was
0.97 times the bound, and with `k: 4` it was 0.79 times the bound. Candidates that keep the
probabilities in float32 (float16) stayed below 0.34 (0.43) of the default bound, even when split.

At `k: 4` the bound still rejects the QA controls by a wide margin. An output rounded to FP8 peaks
at 8.3 times the bound, with 58-70% of elements out of bounds. A 16-ulp scaling peaks at 10 times
the bound, with 87% out of bounds. One case is also rejected: a candidate that rounds its split-K
partial outputs to bfloat16 before merging them reached 1.25 times the bound. Keep partials in
float32.
