# Clamped SiLU-and-mul, bfloat16, gate/up width 1536 (swiglu limit 7)

The activation between the two FFN GEMMs of SwiGLU-limit models (GPT-OSS, DeepSeek-V4-style): the
fused gate/up projection output `x` has shape `[n_tokens, 3072]` (gate in the first 1536 columns,
up in the last 1536, as `chunk(2, dim=-1)`), and
`out = silu(min(gate, 7)) * clamp(up, -7, 7)` is written as `[n_tokens, 1536]` bfloat16.
Arithmetic is float32; the reference is float64.

- **Production origin.** AITER `fused_clamp_act_mul` (`aiter/ops/triton/fusions/fused_clamp_act_mul.py`);
  widths 2048 and 3072 are the shapes of AITER's own test, and this task uses D = 3072 (N = 1536).
- **Baseline.** The AITER Triton device kernel vendored at AITER commit 652f2d4b (MIT, `baseline/LICENSE`),
  one program per token row. The FP8/MXFP8 output branches, the activation selector and the optional
  per-row/per-element weight multiply are removed (this scenario is SiLU, no weights), and the logger is
  gone; the kernel body is otherwise unchanged. Launch configuration (block = next power of two of N,
  warps by block size) is AITER's dispatch.
- **Input distribution.** `x ~ N(0, 4^2)`, not unit normal: with unit variance a limit of 7 would never
  clamp. At std 4 about 8% of gate and up values exceed the limit, so both clamps are exercised.
- **Workloads.** Prefill token counts (5120 to 524288 tokens, 102x span). Each performance case moves
  9216 B per token, at least 10 us estimated at 5120 tokens. Decode batches (1, 3, 48) and the unaligned
  1000-token case are correctness-only. The largest case needs about 3.2 GB of input and a 6.4 GB float64
  reference (the reference is evaluated in row chunks to bound temporaries).
- **No-limit cases.** `t1_nolimit` and `t1000_nolimit` set `swiglu_limit` to 0, which AITER defines as
  clamping disabled (the kernel's `HAVE_SWIGLU_CLAMP=False` path); the reference applies no clamp then.
  AITER's other test widths (2048) and weight-multiply modes are separate scenarios not covered here.
- **Comparison.** Default `baseline_relative`; no override.
