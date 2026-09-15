# mi355x-sg-mxfp8-grouped-gemm

SGLang `_mxfp8_grouped_gemm_kernel` (fused-MoE grouped GEMM via `tl.dot_scaled`) image_kernel
harness, verified on MI355X/gfx950.

- **Source**: `srt/layers/moe/moe_runner/triton_utils/mxfp8_moe_amd_gfx95.py`
- **Target kernel**: `_mxfp8_grouped_gemm_kernel`; timed launcher `_grouped_gemm_mxfp8`.
  The harness builds one MoE forward (MoE-align + activation quant + SwiGLU-OAI, untimed)
  and times the two grouped-GEMM launches (GEMM1 `a_div=top_k`, GEMM2 `a_div=1` weighted).
- **Shapes**: real MiniMax-M3-MXFP8 (TP=8) MoE dims — hidden 6144, per-rank inter 384,
  128 experts, top-k 4 — across decode (T=1, 64) and prefill (T=16384). MoE routing is
  value-dependent so the live GEAK capture produced no oracle; dims are derived from model
  `config.json` (intermediate 3072 / TP8 → 384) and the session's linear-kernel regime.
  See `session_cases.json`.
- **Dtype**: FP8-E4M3 operands, UE8M0 uint8 per-1×32 block scales, FP32 accumulate, BF16 output.
- **Timing**: CUDA-graph replay of the two grouped GEMMs (device time; excludes host launch overhead).
- **Note**: correctness caps T at 64 (the O(T·top_k) reference), performance uses the full token counts.

Run: `python3 scripts/task_runner.py {compile,correctness,performance}`.

## Effective task instructions

Optimize the SGLang fused-MoE grouped GEMM kernel _mxfp8_grouped_gemm_kernel (tl.dot_scaled, CDNA4/gfx950) and its launcher _grouped_gemm_mxfp8 in srt/layers/moe/moe_runner/triton_utils/mxfp8_moe_amd_gfx95.py. The harness times both grouped-GEMM launches of one MoE forward (GEMM1 a_div=top_k, GEMM2 a_div=1 weighted). Cases are the real MiniMax-M3-MXFP8 (TP=8) MoE dims: hidden 6144, per-rank inter 384, 128 experts, top-k 4, across decode (T=1,64) and prefill (T=16384); see session_cases.json. MXFP8 contract: FP8-E4M3 operands, UE8M0 uint8 per-1x32 block scales, FP32 accumulate, BF16 output, SwiGLU-OAI activation. Preserve all correctness cases and improve the CUDA-graph measured performance.

## Arena v2 contract

The candidate is the existing implementation in the declared pinned sources.
Its required final language and exact task-relative editable files are in
`config.yaml`; directory names do not select execution behavior. The framework
freezes this initial implementation into a separate baseline workspace. Both
roles run the same protected harness in their own workspace; an absent candidate
or missing image source is an error, never permission to use the installed copy.

Setup stages the declared Git package, then runs `python3 scripts/setup_task.py`
before baseline capture. It validates source paths and required build assets.
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

## Pinned implementation acquisition

`workspace.sources` acquires SGLang commit
`3ea875fef48f6f01fa3bddd9e2197ad190cef29d` from the declared upstream Git
repository. Setup copies its `python/sglang` package into the existing `sglang/`
candidate layout before the framework freezes the baseline. It performs no
additional download and refuses to overwrite an existing candidate. Both MXFP8
implementation files at this commit are byte-identical to the files inventoried
in the previously qualified source image; their image and source SHA-256 evidence
is recorded in the suite README.

This makes source acquisition independent of whether the scoring image still
ships these modules. Torch, Triton and other runtime dependencies still require
GPU qualification. The numerical reference, shapes, seeds, tolerances and timing
methodology are unchanged. Do not replace the implementation with a similarly
named AITER or vLLM operator.
