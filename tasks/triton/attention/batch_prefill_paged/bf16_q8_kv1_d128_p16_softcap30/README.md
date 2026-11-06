# Paged batch prefill attention, soft cap 30 (8 query heads, 1 KV head, head 128, page 16)

Causal attention of a prefill (extend) batch over a paged KV cache with tanh logits soft-capping:
`s = logits_soft_cap * tanh(softmax_scale * q.k / logits_soft_cap)`, `out = softmax(s) v`, rounded to
bfloat16. Eight query heads share one KV head. Each sequence's new query tokens are packed in `q` at
`cu_seqlens_q`. Its keys and values (a cached prefix, possibly empty, then the new tokens) live in the
pages `kv_page_indices[kv_indptr[b]:kv_indptr[b+1]]` of `k_cache`/`v_cache`
(`[pages, page_size, kv_heads, head_size]`). Every page is full except the last, which holds
`kv_last_page_lens[b]` tokens. The causal mask is bottom-right aligned: query row `r` of a sequence with
`qo_len` queries and `kv_len` keys sees key `c` iff `c <= kv_len - qo_len + r`.

**Production origin.** This is AITER's Composable Kernel `mha_batch_prefill`
(`aiter.mha_batch_prefill_func`, `csrc/py_itfs_ck/mha_batch_prefill_kernels.cu`). SGLang's AITER attention
backend calls it for extend/prefill with `causal=True` and the layer's `logit_cap`. The scenario fixes the
Grok-family attention configuration: SGLang `grok.py` uses a soft cap of 30 and head size 128, and Grok-2's
64 query / 8 KV heads at TP8 leave 8:1 per GPU. The scenario also fixes the 16-token page and bfloat16 KV
cache of the retired `mi300x_sglang_hip_mha_batch_prefill` task. That task used the same configuration,
which also mirrors AITER's `op_tests/test_batch_prefill.py`.

**Scenario and workloads.** Each sequence appends `qo` new tokens to a cached prefix and attends to all
`kv = prefix + qo` keys, as SGLang's extend path does for chunked prefill and prefix-cache hits. A fresh
prompt has no prefix (`kv == qo`). The var axes are the batch and the batch's mean query and key lengths,
`qo_len` and `kv_len >= qo_len`. `inputs.py` lengthens and shortens the query lengths in pairs by
`qo_len/8`, `2*qo_len/8` and `3*qo_len/8` in turn, and shortens and lengthens the cached prefixes in pairs
by `prefix/8`, `3*prefix/8` and `2*prefix/8` (`prefix = kv_len - qo_len`). A batch therefore mixes
prompt and prefix lengths, as a real batch does, and the ragged `cu_seqlens_q`, `kv_indptr` and
bottom-right causal offsets are exercised. The lengths are deterministic per case, so both input draws do
the same work. The page pool has `batch * (ceil(kv_len / 16) + 1)` pages. `kv_page_indices` is a random
permutation of the pool, and its first `kv_indptr[batch]` entries are used, so pages are scattered as in
a long-running server, and the pool's spare pages and the unused slots of every last page hold finite
data that must not be read into the result.

The performance cases range from a single 1k-token prompt to Grok-2's 128k maximum context, unchunked,
plus batches that fill 8k-16k-token prefill budgets, an 8k chunk of a long prompt with 24k tokens
already cached, and 512-token follow-up turns on about 3.5k cached tokens. Short prompts, short extends
and small batches finish below the 10 µs timing floor and are correctness-only. Among them is the
retired task's batch-4 × 512 case, which does as much attention work as a single 1k prompt but spreads
it over four sequences. The retired task's batch-2 × 2048 case is kept as a performance case, and its
48-new-token, 80-key correctness case is kept as `c_b1_q48_kv80` (with this scenario's 8:1 heads; the
retired task's 4:2-head variant is a different scenario). Further correctness cases cover a single new
token per sequence on a cached prefix, one page of new tokens on a 4k prefix, AITER's
`qo_len = 1023, kv_len = 1024` test shape and the maximum batch with and without prefixes.

**Input distribution.** `q` and `k_cache` are drawn with standard deviation 2, so the scaled logits have
standard deviation about 4 and the largest ones reach the cap's nonlinear range (for example
`30 * tanh(17 / 30) = 15.4`). With unit-variance inputs the logits stay near ±1, where the cap is almost
the identity, and a kernel that skipped it would pass.

**Baseline.** The CK kernel cannot be vendored as source, so the baseline is a Triton rewrite with the same
semantics. It is built from AITER's Triton `kernel_unified_attention_2d` (itself adapted from vLLM) and
its `select_2d_config`. The 2D block table is replaced by the CK interface's page list. Each sequence's
key length comes from its page count and last-page length, and the page-list load is masked so that it
never reads past a sequence's pages. Production passes `max_seqlen_q`. The baseline instead uses the
mean query length to choose between the short and long prefill configurations, so the host never reads
device data. Unused features (ALiBi, QQ bias, sinks, FP8, shuffled cache, sliding window, all-decode,
the 3D split-KV path) are removed. The baseline is slower than the production CK kernel, and in
particular it does not split long sequences over several workgroups when the batch is small.

**Comparison.** The default `baseline_relative` bound.
