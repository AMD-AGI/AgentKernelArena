# mi355x-kimi-k2.7-code-fused-moe-gptq-awq-20260724

Self-contained image_kernel harness for the vLLM Triton WNA16 (int4 weight /
bf16 activation) fused-MoE kernel `fused_moe_kernel_gptq_awq`
(`vllm/model_executor/layers/fused_moe/fused_moe.py`).

Generated from the Kimi-K2.7-Code Hyperloom 2026-07-24 MI355X session
(`0e13b6b5-a63f-44e2-b6ff-cc2308e6cb82`), where two same-named leaf sequences of
this kernel were the largest compute leaves (43.862% and 30.391% of E2E). It is
the GPTQ/AWQ int4 weight-only expert GEMM path; on ROCm it is *always* selected
for int4 MoE because `should_moe_wna16_use_cuda()` requires
`current_platform.is_cuda()` (false on ROCm), so the Triton kernel runs instead of
the CUDA `moe_wna16_gemm`.

Real Kimi-K2.7-Code compressed-tensors config is used: 384 experts, top-8, hidden
7168, per-rank intermediate 256 (moe_intermediate_size 2048 under TP=8), int4
symmetric (zero-point=8) with group_size=32 (matching the observed weight/scale
shapes w1 (384,512,3584), w2 (384,7168,128), scale (384,7168,8)). Weights are
synthesized as random int4 with per-group bf16 scales; both the kernel and the
reference use the identical dequantized weights, so correctness is exact up to
bf16 rounding. See `session_cases.json` for full provenance.

The kernel is loaded from the editable workspace copy of the in-image source tree
(custom-op registration is suppressed during load so it does not clash with the
installed copy), so agent edits to `fused_moe.py` take effect (Triton JIT
recompiles on source change).

Expected runtime image:

```text
harbor.crusoe.primus-safe.amd.com/sync/vllm-openai-rocm:v0.24.0
```
