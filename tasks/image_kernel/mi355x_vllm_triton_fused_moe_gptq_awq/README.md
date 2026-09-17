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

## Effective task instructions

Optimize the Triton WNA16 (int4 weight / bf16 activation) fused-MoE kernel fused_moe_kernel_gptq_awq (in fused_moe.py) on MI355X/gfx950. This is the GPTQ/AWQ-quantized expert GEMM path; on ROCm it is always selected for int4 weight-only MoE (should_moe_wna16_use_cuda requires is_cuda(), false on ROCm), so it dominates Kimi-K2.7-Code's MoE. The harness cases use the real Kimi-K2.7-Code compressed-tensors config (384 experts, top-8, hidden 7168, per-rank intermediate 256, int4 symmetric group_size=32) reconstructed from the Hyperloom 2026-07-24 session and stored in session_cases.json. Preserve all correctness cases and improve CUDA-graph measured performance. Do not change the signature of fused_moe_kernel_gptq_awq, invoke_fused_moe_wna16_triton_kernel or fused_experts_impl.

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
