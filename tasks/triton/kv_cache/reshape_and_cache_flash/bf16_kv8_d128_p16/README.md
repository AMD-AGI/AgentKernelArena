# reshape_and_cache_flash, bfloat16, 8 KV heads of 128, block size 16

Writes the key and value vector of every token into paged KV caches in NHD layout
(`[num_blocks, block_size, num_kv_heads, head_size]`) at the slot given by `slot_mapping`; slot `-1`
marks a padding token and is skipped. The caches are mutable inputs: the evaluator restores their
initial (random) contents before every call and checks the whole cache, so a write to a wrong or
untouched slot is detected. The operator is a pure bfloat16 copy, so both caches are compared
`exact` (bitwise) against the float64 reference; the default error bound would be meaningless for a
copy. This departure from the default bound is intentional.

- Production origin: vLLM `triton_reshape_and_cache_flash` (`vllm/v1/attention/ops/`), called for every
  attention layer on each step to append the new K/V to the cache. `slot_mapping` follows the block
  allocator: tokens fill blocks in order, the block ids are a random permutation of the pool (the
  pool holds 25% more blocks than the batch needs), the last 1/32 of the tokens is graph padding and
  about 1% of the remaining slots are skipped.
- Baseline: the vLLM Triton kernel (one program per token) vendored at the recorded revision. The FP8
  cache path with its scale tensors, the head-major key-cache layout and the platform checks are
  removed because this scenario is a bfloat16 NHD cache; the launch configuration is vLLM's ROCm one.
- Workloads: token counts per step from decode (1 token) to batched prefill (409600 tokens); counts
  below 4096 take less than the 10 us timing floor and are correctness-only (the operator moves 8 KB
  per token). Timing is tight at 4096 tokens (about 11 us estimated).
