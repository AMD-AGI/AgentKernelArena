# mi355x-sg-mxfp8-linear

SGLang `_mxfp8_linear_kernel` (dense MXFP8 GEMM via `tl.dot_scaled`) image_kernel harness,
verified on MI355X/gfx950.

- **Source**: `kernels/ops/quantization/mxfp8_amd_gfx95.py`
- **Target kernel**: `_mxfp8_linear_kernel`; timed launcher `_run_mxfp8_linear_kernel`
  (inner GEMM only — excludes the separate activation-quant kernel, matching the profiled leaf).
- **Shapes**: real MiniMax-M3-MXFP8 (TP=8) `qkv_proj` (N=1280, K=6144) and `o_proj`
  (N=6144, K=1024) families across decode (M=1, 64) and prefill (M=16384).
  Recovered from session `17520246` GEAK capture (`_mxfp8_linear_kernel_task/meta.json`)
  and model `config.json`; see `session_cases.json`.
- **Dtype**: FP8-E4M3 operands, UE8M0 uint8 per-1×32 block scales, FP32 accumulate, BF16 output.
- **Timing**: CUDA-graph replay (device time; excludes host launch overhead).

Run: `python3 scripts/task_runner.py {compile,correctness,performance}`.
