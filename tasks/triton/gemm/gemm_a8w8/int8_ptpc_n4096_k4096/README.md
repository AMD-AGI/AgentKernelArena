# W8A8 int8 GEMM (per-token x per-channel), N=4096, K=4096, float16 output with bias

`out = (a @ w^T) * (scale_a * scale_b^T) + bias`. `a` holds int8 activation codes `[m, 4096]` with one
float32 scale per row (`scale_a` `[m, 1]`); `w` holds the int8 `[4096, 4096]` weight in the `nn.Linear`
layout with one float32 scale per output channel (`scale_b` `[4096, 1]`); `bias` is float16 `[4096]`.
The int8 products are accumulated exactly in int32, converted to float32, scaled, rounded to float16 and
only then added to the float16 bias (the order of the vLLM kernel). The reference is the same expression in
float64 (the integer sum is exact there) with the bias added in float64, so the baseline's intermediate
float16 rounding shows up in its own error, which `baseline_relative` accounts for.

## Production origin and workloads

This is vLLM's `triton_scaled_mm`, the Triton fallback of the compressed-tensors W8A8 int8 linear scheme
(`CompressedTensorsW8A8Int8`) on ROCm, with per-token activation scales and per-channel weight scales.
The old task (`triton_scaled_mm`) used tiny synthetic shapes (M, N, K <= 512) and several scale/bias
variants; the scenario here fixes the production-style vector scales, a bias and a 4096x4096 layer
(hidden size 4096). Cases span `m = 1 .. 65536`: decode batches and chunks below 512 tokens are
correctness-only, `m = 512 .. 65536` are performance cases (128x span). `m = 65536` is a large batched
prefill; the largest case needs about 1 GB for the inputs and outputs and about 5 GB for the float64
reference. Boundary correctness cases: `m = 1`, `m = 3` (not a power of two) and `m = 17`, `100`, `1000`
(not multiples of 16).

**Inputs.** `inputs.py` quantizes normally distributed activations and weights row-wise (scale = row
amax / 127, round to nearest, clamp to +-127), so codes and scales are consistent; the bias is standard
normal float16.

**Dropped variants.** The old task exercised the kernel on float16 operands with float32 accumulation, scalar
scales and no-bias variants. The compressed-tensors W8A8 scheme only dispatches int8 (or fp8) operands with vector
scales, so only that production path is kept; the scalar-scale branch is not in the baseline. Rows are masked
in the kernel, so the unaligned and tiny-m correctness cases cover its edge handling.

## Baseline

`baseline/kernel.py` is vLLM's `scaled_mm_kernel` (unchanged, Apache-2.0) and the heuristic tile selection of
`triton_scaled_mm` (`use_heuristic=True`: 64x64x256 / 64x128x256 / 64x128x128 for small M, 128x128x128
otherwise), wrapped in `run()`. The argument asserts, the scalar-scale branch (not used with vector scales)
and the trailing `.to(out_dtype)` (a no-op copy) were dropped; the weight is the transposed view of
the `[n, k]` tensor, as in production.

## Estimated timing

Performance cases start at `m = 512` (about 17 GFLOP, estimated above 10 us); smaller shapes are expected
to be below the 10 us floor and are correctness-only. Not measured on GPU.

## Risks for the GPU audit

`c += bias` runs in float16 inside the kernel; the bound is judged against the baseline's own error, so
this is expected to pass but has not been audited. Output magnitudes (about 64 standard deviations
per element before bias) are far from the float16 range limit.
