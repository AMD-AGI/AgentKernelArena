# merge_attn_states, bfloat16, 32 heads, head size 128

Combines the attention outputs of two disjoint key partitions (a cached prefix and the new suffix, or
two split-KV chunks) into the output of attending over all keys, using each partition's log-sum-exp
(section 2.2 of arXiv 2501.01005). The merged log-sum-exp is returned too, so merges can be chained.
Each token costs 24 KiB of traffic (two 8 KiB inputs read, 8 KiB written), so the kernel is bandwidth
bound.

**Production origin.** vLLM's `merge_attn_states` Triton kernel (prefix caching with chunked prefill,
cascade attention, context-parallel attention). SGLang's `merge_state` is the same operator with an
`[tokens, heads]` log-sum-exp layout; this task uses vLLM's `[heads, tokens]` layout.

**Inputs.** `inputs.py` draws log-sum-exp values from N(0, 4) and marks 1/32 of the entries as an empty
prefix (+inf, the FlashAttention-2 encoding) and 1/32 as an empty suffix (-inf, FlashAttention-3). The
other partition is always populated, so outputs and merged log-sum-exp stay finite. Another 1/32 each use extreme finite values (prefix +1000, prefix -1000, both +1000), as the previous
task's controls did. The reference maps +inf to -inf before the maximum, as the kernel does. Both
partitions empty is undefined (the kernel emits 0 and -inf, the reference NaN) and is not generated.
bfloat16 is the fixed scenario dtype; the previous tasks also ran float16 and float32 and other head
configurations, which would be separate scenarios.

**Workloads.** Token counts from decode batches to large prefill batches. Performance cases start at 1536
tokens (about 38 MB of traffic); the 100x span needs the largest case, 196608 tokens, which is a
large offline batch rather than a typical serving chunk. Smaller counts are correctness-only.

**Baseline.** vLLM's Triton kernel, vendored. The optional FP8 output scaling and the
`mask_empty_context` helper were removed; the kernel is otherwise unmodified, including its
`prefill_tokens_with_context` argument, which `run()` sets to the token count.
