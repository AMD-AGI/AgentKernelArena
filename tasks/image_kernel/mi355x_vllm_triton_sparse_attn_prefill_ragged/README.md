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

## Effective task instructions

Optimize the Triton ragged sparse-attention prefill kernel _sparse_attn_prefill_ragged_kernel (in rocm_aiter_mla_sparse.py) on MI355X/gfx950. This is DeepSeek-V4 Sparse Attention (DSA) prefill - each query attends only to a top-k selected set of MLA-latent KV positions supplied in ragged CSR form (indices/indptr); the latent (head_dim=512 = 448 NoPE + 64 RoPE) serves as both K and V. The harness cases use the real DeepSeek-V4-Flash config (64 query heads, index_topk=512, BF16) reconstructed from the Hyperloom 2026-07-24 session and stored in session_cases.json. Preserve all correctness cases and improve CUDA-graph measured performance. Do not change the signature of _sparse_attn_prefill_ragged_kernel or _rocm_sparse_attn_prefill_ragged_triton.

## Arena v2 contract

The candidate is the existing implementation in the declared image sources.
Its required final language and exact task-relative editable files are in
`config.yaml`; directory names do not select execution behavior. The framework
freezes this initial implementation into a separate baseline workspace. Both
roles run the same protected harness in their own workspace; an absent candidate
or missing image source is an error, never permission to use the installed copy.

Setup runs `python3 scripts/setup_task.py` after declared image materialization
and before baseline capture. It validates source paths and required build assets.
Do not edit `scripts/`, workload files or references. Additional source files
outside `candidate.editable` are dependencies, not editable implementation.
Preserve the original numerical gates, seeds, layouts, dispatch, state handling
and CUDA graph/event timing. `workloads.json` enumerates the complete manifest
independently of reported timings; `session_cases.json`, when present, retains
its original session provenance. Cases marked correctness-only are not scored.

Use the agent-neutral commands:

```bash
python3 scripts/evaluate.py validate-task
python3 scripts/evaluate.py baseline compile
python3 scripts/evaluate.py baseline correctness
python3 scripts/evaluate.py baseline performance
python3 scripts/evaluate.py candidate compile
python3 scripts/evaluate.py candidate correctness
python3 scripts/evaluate.py candidate performance
```

Baseline commands run in the framework's frozen workspace. Each command emits
one `ARENA_EVAL_RESULT=` envelope. A failed dependency, dispatch or output contract
is a failure, not an accepted baseline numerical diagnostic. The original
`task_runner.py` remains the protected operator implementation of these checks;
its generated performance region must be materialized by Arena. Optional
profiling does not supply final evaluation evidence. Agent CLI adaptation belongs
to the agent integration; use the declared v2 runner for task evaluation, with
the task's full numerical and workload checks.
This migration has CPU regression coverage; formal GPU task validation and the
optimization campaign are coordinated separately. Runtime source availability
must be checked against the selected immutable image, not inferred from a tag.
