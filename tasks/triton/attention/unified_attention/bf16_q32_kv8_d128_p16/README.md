# Unified paged attention (vLLM), bfloat16, 32 query / 8 KV heads, head size 128, block size 16

Causal attention of mixed decode and chunked-prefill batches over vLLM's paged KV cache. Each sequence has
`q_len` new query tokens (1 for decode) and `seqused_k[s]` KV tokens including them; the query tokens attend
causally to everything at or before their absolute position. Scale is `head_dim ** -0.5`, scores and
accumulation are float32, the output is bfloat16.

**Production origin.** vLLM's Triton attention backend (`TritonAttentionImpl` calling `unified_attention`
in `vllm/v1/attention/ops/triton_unified_attention.py`), the attention path of ROCm vLLM for head sizes the
custom paged-attention kernel does not take and the default of the `TRITON_ATTN` backend. The geometry (32
query heads over 8 KV heads, head size 128, 16-token pages) is the grouped-query layout of the Llama-3-8B
family. It replaces the three old pieces of this operator (`triton2triton/vllm/triton_unified_attention_2d`,
`_3d`, `triton_reduce_segments`), which tested an older vLLM that kept the 2D kernel, the 3D kernel and the
segment reduction as separate entry points; the vendored revision merges the first two into one kernel
(`IS_3D`) and keeps `reduce_segments`.

**Scenario and workloads.** One scenario: bfloat16 query and cache, no sliding window, softcap, sinks or fp8.
The KV cache is vLLM's packed layout `[num_blocks, block_size, kv_heads, 2 * head_dim]`, key in the first and
value in the second half of the last dimension; the baseline splits it into two strided views, as
`TritonAttentionImpl` does. The varying axes are `num_seqs`, `q_len` and `ctx_len` (the longest KV length).
The generator (`inputs.py`) draws ragged KV lengths (the longest sequence has `ctx_len`, the others between
`ctx_len / 2` and `ctx_len`, never fewer than `q_len`) and scatters the pages of all sequences over the pool
with a random permutation. Performance cases cover both launch paths: single-sequence and small-batch decode
at 4K to 32K tokens (up to 16 sequences: the segmented 3D kernel with its reduction) and decode batches of 64
and 256 sequences (2D kernel), plus prefill chunks of 2048 to 8192 query tokens. The shapes are representative
serving shapes, not recorded from a trace; the one-above-threshold and context-512/513 decode cases and the
small ragged prefill shapes are correctness-only.

**Baseline.** vLLM at `edbc4969`, vendored: the unified kernel, `reduce_segments` and the shared device helpers
(`helpers.py`) are unchanged, as are the dispatch decisions of `unified_attention` (tile sizes by prefill or
decode, 2D unless the batch is decode-only with at most `seq_threshold_3D` sequences). The values
`TritonAttentionImpl` supplies are reproduced: 16 parallel softmax segments and `seq_threshold_3D = 128 //
num_kv_heads` (16; vLLM additionally snaps it to the nearest CUDA-graph capture size, which is 16 as well).
Features outside the scenario (sinks, alibi, qq-bias, softcap, multimodal prefix, R-SWA, chunked attention,
fp8 and int4 caches, Intel tensor descriptors, the B200 large-head tuning) are not passed on; the kernel keeps
their constexpr branches. `max_seqlen_q` is `q.shape[0] // num_seqs`. vLLM keeps the segment buffers as persistent
workspace; they are allocated inside `run()`, which inside a CUDA graph costs nothing per replay.

**Comparison.** Default `baseline_relative` on the bfloat16 output.
