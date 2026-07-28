# mi355x-deepseek-v4-flash-sparse-attn-prefill-20260724

Self-contained image_kernel harness for the vLLM Triton DeepSeek-V4 sparse-attention
prefill kernel `_sparse_attn_prefill_ragged_kernel`
(`vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`).

Generated from the DeepSeek-V4-Flash Hyperloom 2026-07-24 MI355X session
(`5fa5a97c-fbf2-4c3e-a84e-78576f745622`), where this leaf was 10.231% of the
timeline wall. It implements DeepSeek Sparse Attention (DSA) prefill: each query
attends only to a top-k selected set of MLA-latent KV positions supplied in ragged
CSR form (`indices` / `indptr`). The latent (head_dim=512 = 448 NoPE + 64 RoPE) is
used as BOTH K and V; the kernel runs an online softmax over the selected positions.

Real DeepSeek-V4-Flash config is used: 64 query heads, single latent KV head,
head_dim 512, `index_topk`=512, BF16. The trace records this leaf as a
graph-synthetic op with empty input dims, so `sq` (prefill query count) uses the
model's observed token buckets and `num_kv` (KV pool) is a representative
reconstruction (>= topk); the kernel is pure ragged gather-attention, so
correctness is well-defined for any synthesized sparse pattern. See
`session_cases.json` for full provenance.

The kernel is loaded from the editable workspace copy of the in-image source tree,
so agent edits to `rocm_aiter_mla_sparse.py` take effect (Triton JIT recompiles on
source change).

Expected runtime image:

```text
harbor.crusoe.primus-safe.amd.com/sync/vllm-openai-rocm:v0.24.0
```
