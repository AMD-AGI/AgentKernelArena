# Fused residual add + LayerNorm, bfloat16, hidden size 4096

Adds `residual` to `x`, stores the sum rounded to bfloat16 as `residual_out`, and applies an affine
LayerNorm (mean and biased variance over the hidden axis) to the stored sum, writing `out`. The reference
normalizes the bfloat16-rounded sum, as the operator does, and computes in float64.

Baseline: AITER's Triton `layernorm2d_fwd_with_add` (ROCm/aiter at the recorded revision), one program per
row. It replaces the Composable Kernel `layernorm2d_fwd_with_add` used by the previous version of this task,
which cannot be vendored as source. The kernel is copied unchanged; the launcher is inlined into `run()`.

Workloads: token counts at hidden size 4096 (the other width of the previous task). Decode-sized batches run faster
than the 10 us timing floor and are correctness-only; the performance cases (1536 to 196608 tokens, 128x)
are prefill chunks and batched prefill. The upper end is bounded by the float64 reference memory. The
weight and bias distributions (uniform 0.5 to 1.5, normal with std 0.1) stand in for trained parameters.
Epsilon 1e-5 is the value of the previous task. The previous task also covered hidden size 8192, which is the sibling scenario `bf16_h8192`.
