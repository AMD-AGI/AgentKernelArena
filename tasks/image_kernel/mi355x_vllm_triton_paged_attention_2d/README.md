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
