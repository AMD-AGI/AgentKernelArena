# Sparse MLA prefill with attention sink, DeepSeek-V4 CSA layer (64 heads, head size 512, top-k 512 + window 128)

DeepSeek-V4 sparse-attention prefill for a layer with compression ratio 4: every query token attends to a
selected subset of a latent KV pool, given as ragged CSR lists (`indices`, `indptr`), with a learned per-head
attention sink that adds `exp(attn_sink[h])` to the softmax denominator and contributes no value. The 512-wide
latent row (448 NoPE + 64 RoPE) is both key and value, all 64 query heads share it, and an empty row
yields zeros. This is the same operator as the sibling `bf16_h64_d512_topk512` scenario (vLLM's kernel,
random indices, no sink); this scenario adds the sink and the index structure of a real CSA layer.

## Production origin of the baseline

vLLM's ROCm DeepSeek-V4 prefill with attention sinks: `rocm_sparse_attn_prefill` in
`vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` (`vllm-project/vllm` at
`edbc4969a76b25c531f7b9bb16b79984e65ef023`; header `SPDX-License-Identifier: Apache-2.0` and
`SPDX-FileCopyrightText: Copyright contributors to the vLLM project`, kept verbatim). The production caller,
`vllm/models/deepseek_v4/amd/rocm.py`, builds per-token combined indices (compressed top-k entries followed by the
sliding window) and passes `attn_sink=self.attn_sink`. The function has two routes: dense indices with a length vector
(converted to ragged lists by `build_ragged_indices_from_dense`, which uses PyTorch reductions) and a ragged
route taking `ragged_indices` and `ragged_indptr`. Both end in the same Triton kernel,
`_sparse_attn_prefill_ragged_kernel`. This task takes the ragged route, because its input is the CSR index list and
the dense conversion is host-side PyTorch work that `run()` may not do; the kernel and its launch are identical.

Changes: only the kernel (byte-identical to upstream) and the launch code of
`_rocm_sparse_attn_prefill_ragged_triton` are vendored. The launch keeps the upstream configuration (`BLOCK_H=16`,
`BLOCK_K=16` for head size 512, 4 warps; no autotune). Upstream allocates a bfloat16 tensor and then calls
`output.copy_(output_chunk.to(output.dtype))`; here the kernel writes the preallocated bfloat16 `out` directly (the
output dtype equals the kernel's, so no conversion is needed). `HAS_ATTN_SINK=True` is always passed.

### Context: AITER's copy of the kernel

`ROCm/aiter` carries a header-less near copy, `_sparse_attn_prefill_kernel` in
`aiter/ops/triton/_triton_kernels/attention/sparse_attention_dsv4.py` (commit `b50066a91`), with `-inf` handling, an
autotune and no license header. It was not vendored; this task uses the vLLM original it matches in structure and names.

## Scenario

bfloat16, 64 query heads, head size 512, `scale = 1/sqrt(512)`, index list of a compression-ratio-4 layer: at most
512 compressed entries plus a 128-token sliding window. Citations: DeepSeek-V4-Flash `config.json` (Hugging Face):
`num_attention_heads` 64, `head_dim` 512, `qk_rope_head_dim` 64 (hence 448 NoPE dimensions), `num_key_value_heads` 1,
`index_topk` 512, `sliding_window` 128, `compress_ratios` (alternating 4 and 128 across the layers; the 4 layers
are the sparse CSA layers modelled here). The attention sink values are model weights that no public config
carries; the task draws them from a standard normal distribution (a synthetic stand-in, visible in the output
through the denominator for logits of order 1). The softmax scale is the same `1/sqrt(512)` as the sibling task (the yarn
attention-scaling factor of the checkpoint's `rope_scaling` is not applied).

## Workloads and inputs

One var axis, `n_queries`: the tokens of one prefill chunk of one prompt starting at position 0 (the chunked
prefill sizes of a server, 128 to 16384; the CSA layers' count of compressed entries is `n_queries // 4`).
`inputs.py` builds the production index structure: token `i` lists `min((i + 1) // 4, 512)` compressed entries (all of them while
there are at most 512, afterwards 512 distinct entries drawn uniformly in increasing order, standing in for
the indexer's top-k), then the last `min(i + 1, 128)` token rows. Rows therefore grow from 1 to 640 entries: the
first rows have no compressed entry at all, the window is partial for the first 127 tokens, and chunks
above 2051 tokens have rows with sampled selections. `q` and the pool are standard normal.

The vLLM launch (16-head blocks, so 4 programs per query row) gives the GPU 4x more programs than a 64-head block. Estimated baseline time of the performance cases: a row moves about 128 KB (q and out) plus its gathered KV rows,
which are shared between neighbouring rows through the cache; 128 tokens take about 9 us at the bandwidth model and, with
64-row-wide head blocks and a serial loop over 5-ish key blocks per program, an estimated 15-40 us on the real
kernel; 16384 tokens move 2.1 GB of q and out and do 1.4 TFLOP, about 1.5-4 ms. The 128 to 16384 range is a 128x
span. The 128-token case is the one closest to the 10 us floor.
Correctness-only cases: 1, 3, 5, 64, 100, 129, 333, 1073, 2049 and 7211 tokens.

## Error bound

Default `baseline_relative`: the dominant error is the bfloat16 rounding of the probabilities in the second matrix product, as
for the sibling task. The reference returns float32 (computed in float64) so the 16384-token case stays within memory.
