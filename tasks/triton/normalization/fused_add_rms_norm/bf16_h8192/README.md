# Fused residual add + RMSNorm, bfloat16, hidden size 8192

Adds `residual` to `x`, stores the sum rounded to bfloat16 as `residual_out`, and writes
`out = residual_out * rsqrt(mean(residual_out^2) + epsilon) * weight`. The reference normalizes the
bfloat16-rounded sum, as the operator does, and computes in float64. The residual update is declared as a
separate output (`residual_out`), matching AITER's `rmsnorm2d_fwd_with_add` signature, so both inputs stay
read-only.

Baseline: AITER's Triton `rmsnorm2d_fwd_with_add` (ROCm/aiter at the recorded revision). The kernel is
copied unchanged; the launcher is inlined into `run()` (persistent programs, one per compute unit). It
is the same kernel that the previous version of this task started from.

Workloads: token counts at hidden size 8192 (a sibling of `bf16_h4096`, covering the 8192 width of the previous tasks). Decode-sized batches
are correctness-only because they run faster than the 10 us timing floor; the performance cases (768 to
98304 tokens) cover prefill chunks and batched prefill. The upper end is bounded by the float64
reference memory. Epsilon 1e-6 and the weight distribution follow the vLLM RMSNorm convention.
