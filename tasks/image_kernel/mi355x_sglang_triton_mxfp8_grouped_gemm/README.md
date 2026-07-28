# mi355x-sg-mxfp8-grouped-gemm

SGLang `_mxfp8_grouped_gemm_kernel` (fused-MoE grouped GEMM via `tl.dot_scaled`) image_kernel
harness, verified on MI355X/gfx950.

- **Source**: `kernels/ops/moe/mxfp8_moe_amd_gfx95.py`
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
