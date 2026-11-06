# Unified paged attention (vLLM), Gemma-4 sliding layer: bfloat16, 8 query / 4 KV heads, head size 256, window 1024

Causal sliding-window attention of mixed decode and chunked-prefill batches over vLLM's paged KV cache. Each
sequence has `q_len` new query tokens (1 for decode) and `seqused_k[s]` KV tokens including them; a query token
attends to the 1024 KV tokens up to and including its own position (vLLM's `window_size=(1023, 0)`, the kernel's
`SLIDING_WINDOW=1024`). Scale is `head_dim ** -0.5`, scores and accumulation are float32, the output is bfloat16.

**Production origin.** The sliding layers of gemma-4-26B-A4B-it served by vLLM's `TRITON_ATTN` backend on MI355X
(production serving trace, TP=2: 8 query heads over 4 KV heads at head size 256, window 1024). The same
`kernel_unified_attention` was the largest single kernel in that trace (19.6% of GPU time at 64 decode sequences).
Replaces `image_kernel/mi355x_vllm_triton_unified_attention_gemma4` for its sliding layers; that task's
full-attention layers (head size 512, one KV head, no window) are a different scenario and are not part of this
task. The AITER implementation of the same operator is a different baseline (`bf16_q12_kv2_d128_p16`).

**Scenario and workloads.** One scenario: bfloat16 query and cache, window 1024, no softcap, sinks or fp8. The KV
cache is vLLM's packed layout `[num_blocks, block_size, kv_heads, 2 * head_dim]`, key in the first and value in
the second half of the last dimension; the baseline splits it into strided views, as `TritonAttentionImpl` does.
The varying axes are `num_seqs`, `q_len` and `ctx_len` (the longest KV length). The generator (`inputs.py`) draws
ragged KV lengths (the longest sequence has `ctx_len`, the others between `ctx_len / 2` and `ctx_len`, never fewer
than `q_len`) and scatters the pages over the pool with a random permutation. Contexts above 1024 tokens make the
window clip, so a decode step reads only the last 1024 tokens of each sequence (about 4 MB), which is why the
decode performance cases use 16 or more sequences. The trace pins the 64-sequence, context-1024 decode (the
`s64_q1_c1024` case); the larger batches, longer contexts and the prefill chunks are representative serving
shapes, not recorded (the trace does not record the per-sequence composition of its prefill steps). The three
smallest and the window-boundary shapes are correctness-only. Up to 32 sequences of decode take vLLM's 3D kernel
(with the window-pruned segments), larger batches and prefill the 2D kernel.

**Baseline.** vLLM at `edbc4969`, vendored: the unified kernel, `reduce_segments` and the shared device helpers
(`helpers.py`) are unchanged, as are the dispatch decisions of `unified_attention`, including the Gemma-3/4
rule that selects a 32-token tile for window 1024 at head size 256. The values `TritonAttentionImpl` supplies are
reproduced: 16 parallel softmax segments and `seq_threshold_3D = 128 // num_kv_heads` (32; vLLM snaps it to the
nearest CUDA-graph capture size, which is 32 as well). Features outside the scenario (sinks, alibi, qq-bias,
softcap, multimodal prefix and its Gemma-4 window clamp, R-SWA, chunked attention, fp8 and int4 caches, Intel
tensor descriptors, the B200 large-head tuning) are not passed on; the kernel keeps their constexpr branches.
`max_seqlen_q` is `q.shape[0] // num_seqs`. The segment buffers vLLM keeps as persistent workspace are allocated
inside `run()`, which inside a CUDA graph costs nothing per replay. The revision is the vLLM checkout of the
repository tooling; the image the trace ran (vLLM 0.24.0) was not available to compare with it.

**Comparison.** Default `baseline_relative` on the bfloat16 output.
