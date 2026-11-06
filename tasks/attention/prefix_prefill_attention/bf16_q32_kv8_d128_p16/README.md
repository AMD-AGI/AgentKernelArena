# Prefix-prefill attention, bfloat16, 32 query / 8 KV heads, head size 128, 16-token pages

Attention for the extend step of chunked prefill and prefix caching: each sequence has `ctx_len`
tokens already in a paged KV cache and `q_len` new tokens. Every new token attends to the whole
cached prefix and causally to the new tokens up to itself (grouped-query attention, group size 4).
The caches use vLLM's layouts (K: `[pages, kv_heads, head_dim/x, page_size, x]` with `x = 8`,
V: `[pages, kv_heads, head_dim, page_size]`) and a block table `b_loc` that maps the logical pages
of every sequence to physical pages in random order. Unused slots of the last page hold random
finite data, so a kernel must mask the context length itself.

**Production origin.** vLLM's `context_attention_fwd` (`vllm/v1/attention/ops/prefix_prefill.py`) for
the Triton attention backend; the shape is a Llama-3-70B-style attention layer
(32 query heads, 8 KV heads, head size 128) with the default 16-token pages. SGLang's
`extend_attention` computes the same operator over a token-level index list; this task uses the vLLM
paged layout, which the scenario name (`p16`) refers to.

**Scenario and workloads.** `batch`, `q_len` and `ctx_len` vary. Sequences of a call have identical
lengths (the packed token count is `batch * q_len`), which lets the launch grid derive the maximum
query length from the tensor shapes. The performance cases are chunked-prefill and prefix-cache
shapes: 64-4096 new tokens over 2k-16k cached tokens at batch 1-8, spanning more than 1000x in
`batch * q_len * ctx_len`. Single-token (decode-like) and tiny extends are correctness-only. The
`sm_scale` scalar is `1/sqrt(128)`.

**Baseline.** vLLM's Triton `_fwd_kernel` (block sizes 128x64 for power-of-two pages, context tile
of 32 tokens, `num_warps=4`, `num_stages=1`), as simplified by the retired
`triton_paged_prefix_prefill` task: the fp8-cache, attention-sink, skip-decode, non-causal and
cached-K/V paths are removed and the sliding window is off. The host wrapper became `run()`.
The old task's other scenarios (head sizes 16-128, non-power-of-two pages, sliding window) and the
Lq=192/Lv=128 MLA variant of the SGLang task are not part of this scenario.

**Reference.** float64 attention that rebuilds the dense context from the caches through the block
table and softmaxes in query chunks. Default comparison bound (`baseline_relative`).

**Risks for the GPU audit.** The smallest performance case (`1 x 64 x 2048`) launches only 32
programs, each looping over 64 context tiles; it is expected well above 10 us but unmeasured.
