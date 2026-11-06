# reshape_and_cache_flash_diffkv, bfloat16, 8 KV heads, K 192 / V 128, block size 16

Like `reshape_and_cache_flash`, but K and V have different head sizes and share one cache
(`[num_blocks, block_size, num_kv_heads, head_size_k + head_size_v]`, each head holding its K vector
followed by its V vector). Slot `-1` marks a padding token and is skipped. The cache is a mutable
input: the evaluator restores its initial (random) contents before every call and checks all of it.
The operator is a pure bfloat16 copy, so the cache is compared `exact` (bitwise) against the
float64 reference; the default error bound would be meaningless for a copy. This departure from the
default bound is intentional.

- Production origin: vLLM `triton_reshape_and_cache_flash_diffkv` (`vllm/v1/attention/ops/`), used by
  the diff-KV attention backends (`flash_attn_diffkv`, `triton_attn_diffkv`) for models whose K head
  is wider than the V head. The head sizes 192/128 and 8 KV heads are a representative GQA scenario
  (K = no-position + rotary parts), not copied from one named model. `slot_mapping` follows the
  block allocator as in the `reshape_and_cache_flash` task (random block permutation, 25% spare
  blocks, graph padding and about 1% skipped slots).
- Baseline: the vLLM Triton kernel (one program per token and head) vendored at the recorded revision,
  minus the FP8 cache path and its scale arguments (this scenario is bfloat16); the launch
  configuration is vLLM's ROCm one.
- Workloads: token counts per step from decode to batched prefill (409600 tokens); counts below 4096
  take less than the 10 us timing floor and are correctness-only (10 KB moved per token). Timing is
  tight at 4096 tokens (about 13 us estimated).
