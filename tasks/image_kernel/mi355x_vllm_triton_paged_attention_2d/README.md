# mi355x-qwen3_5-122b-paged-attention-2d-20260724

Self-contained image_kernel harness for the vLLM Triton decode paged-attention
kernel `kernel_paged_attention_2d`
(`vllm/v1/attention/ops/chunked_prefill_paged_decode.py`).

Generated from the Qwen3.5-122B-A10B-FP8 Hyperloom 2026-07-24 MI355X session
(`90d4b4a8-1db6-4e9b-9ce4-b1d9e7d5238d`), where this leaf was the single largest
GPU kernel at 15.799% of the timeline wall. It is the Triton fallback path: the
hand-written ROCm asm paged-attention only supports head_size 64/128 on gfx9, and
this model uses head_size 256 (GQA 4:1, BF16), so decode falls back to this Triton
kernel.

The kernel is loaded from the editable workspace copy of the in-image source tree,
so agent edits to `chunked_prefill_paged_decode.py` take effect. See
`session_cases.json` for provenance, shapes and dtypes. The per-sequence KV
context length (`ctx_len`) is a representative reconstruction from the model's
decode regime — the trace records the query/output/step shapes but not the KV
context length.

Expected runtime image:

```text
harbor.crusoe.primus-safe.amd.com/sync/vllm-openai-rocm:v0.24.0
```

## Effective task instructions

Optimize the Triton decode paged-attention kernel kernel_paged_attention_2d (in chunked_prefill_paged_decode.py) on MI355X/gfx950. This kernel is the vLLM ROCm fallback path taken when head_size is not 64/128 (here head_size=256, GQA 4:1, BF16), so the fast hand-written ROCm asm paged-attention does not apply and this Triton kernel dominates decode. The harness cases are reconstructed from the Qwen3.5-122B-A10B-FP8 Hyperloom 2026-07-24 session and stored in session_cases.json. Preserve all correctness cases and improve CUDA-graph measured performance. Do not change the public signature of chunked_prefill_paged_decode or kernel_paged_attention_2d.

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
its generated performance region must be materialized by Arena. Developer
profiling drivers do not supply final evaluation evidence.
This migration has CPU regression coverage; formal GPU task validation and the
optimization campaign are coordinated separately. Runtime source availability
must be checked against the selected immutable image, not inferred from a tag.
