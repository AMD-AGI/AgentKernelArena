# Decode attention, bfloat16, 32 query heads / 8 KV heads, head_dim 128

One decode step of grouped-query attention (group size 4): each sequence contributes a single
query token and attends over its whole KV history, which lives in a token-indexed cache (page size 1).
`kv_indptr`/`kv_indices` list each sequence's cache slots, which are scattered across the pool.
`num_kv_splits` is the scheduler's per-sequence split hint from SGLang's backend heuristic
(at most 8 splits, 256 CUs); the generator reproduces that heuristic. An implementation may use or
ignore it, since the result must not depend on it.

- **Baseline:** SGLang's two-stage split-KV Triton decode attention: the grouped stage-1 kernel
  (partial softmax-V and LSE per query-head tile and KV split, MFMA dots) followed by the stage-2
  log-sum-exp merge. It was vendored from `python/sglang/kernels/ops/attention/decode_attention.py`
  and reduced to this scenario's path: the MLA/RoPE-split, PDL, score_mod, sink, logit-cap,
  xai-temperature, forced-split and gfx1250 branches and page sizes above 1 were removed. The
  per-split scratch buffers (`attn_logits`, `attn_lse`) are allocated inside `run()`, so the
  baseline and any candidate pay the same allocation boundary; captured graphs make it free.
  An earlier version of this task used the MHA stage-1 kernel for MHA shapes and fixed 8 splits;
  this scenario is GQA only, with the scheduler's splits as an input.
- **Scenario:** `max_ctx` is the longest sequence; the others are uniformly ragged between half and
  all of it. `pool = batch * max_ctx` slots exist, `kv_indices` is a random permutation of them
  (only the first `kv_indptr[-1]` entries are read), and keys and values are standard normal, so
  scaled scores have unit variance.
- **Workloads:** single long-context streams through batches of 128 with 4k-16k contexts (up to
  2^21 cache slots, 4.3 GB each for K and V; the float64 reference gathers one sequence at a time).
  Short contexts (1-2050) are correctness-only; the smallest performance case (one stream, 16384
  tokens, about 50 MB of KV) is near the latency-bound regime and must be confirmed above 10 us on MI355X.
- **Bounds:** default `baseline_relative`; the bf16 probability cast before the P.V dot and the
  split-wise merge order are part of the baseline's error budget.
