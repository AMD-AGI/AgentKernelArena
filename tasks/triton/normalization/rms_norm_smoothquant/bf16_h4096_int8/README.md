# RMSNorm with SmoothQuant and per-token INT8 quantization, bfloat16, hidden size 4096

The activation quantization in front of an INT8 SmoothQuant GEMM: each bfloat16 token row is RMS-normalized, scaled by
the per-channel RMSNorm weight and by the per-channel smoothing scale `x_scale`, and quantized to INT8 with one
dynamic scale per token. `hidden = x * rsqrt(mean(x^2) + epsilon) * weight * x_scale` stays in float32 (no rounding to
bfloat16 in between, as in the fused production kernels), `y_scale = max|hidden| / 127` and the codes are
`round(hidden / y_scale)`. `epsilon` is 1e-5. The reference is float64 end to end.

## Rounding rule and exact comparison

The framework compares integer outputs bitwise, and the INT8 codes cannot be compared bitwise under truncation: the
row maximum quotients to exactly 127, and `127 - 1 ulp` truncates to 126, so equally valid float32 implementations
(division or multiplication by the reciprocal) would differ on the maximum of every row. The operator is therefore
specified with round-to-nearest (halves away from zero), which is how INT8 dynamic quantization is usually written
(the maximum rounds to 127 either way). The old task truncated toward zero (distance-1 tolerance); this changes the
rule and makes the comparison exact.

To keep the exact comparison decided by the specification and not by float32 noise, `inputs.py` generates `x`: the
reciprocal RMS cancels in `hidden / y_scale`, so the quotient is `127 * p / max|p|` with `p = x * weight * x_scale`,
exact in float64. The generator moves every element within 2e-3 of a rounding decision (a half integer) to the next
bfloat16 neighbour of larger magnitude and repeats until none is left; an implementation's relative error of a few
1e-7 moves the quotient by under 4e-4. Checked on CPU against float32 pipelines (reciprocal multiply and true divide,
RMS perturbed by 1e-6): no code differed on a 2048-token case. `out` uses `exact` for this reason.

## Scale bound

`y_scale` is compared with `tolerance` (`atol` 0, `rtol` 1e-6) rather than `baseline_relative`. The baseline's float32
scale is within about one float32 ulp of the float64 reference, so the default bound (a multiple of the baseline's own
per-row error) is about an ulp and rejects correct float32 candidates: the QA calibration probe's float32 PyTorch
evaluation of the reference failed it in the six cases from 1000 tokens up (violation fraction about 1e-3, peak 1.0x to
1.5x), purely from float32 rounding of the squared sum, `rsqrt`, the products and the division. The measured relative
error of float32 pipelines against the float64 scale (CPU, 8192 tokens, true `rsqrt` or `1/sqrt`) has a maximum of
2.4e-7 and a mean of 5.4e-8; perturbing the reciprocal RMS by two further ulp (a loose hardware `rsqrt`) gives a maximum of
3.7e-7. `rtol` 1e-6 (about 8 ulp) accepts these with a margin of 2.7x and still rejects wrong scales: a divisor of 126
or 128 is off by 8e-3, a bfloat16-rounded scale by up to 4e-3, and a scale from a partial row maximum by far more. The
controls are rejected by the exact `out` comparison regardless of this bound.

## Baseline

AITER's Triton `_quant_rms_norm_kernel` and `_per_token_quant` (`aiter/ops/triton/_triton_kernels/normalization/rmsnorm.py`)
with the launcher `rmsnorm2d_fwd_with_smoothquant`, ROCm/aiter commit `652f2d4b768239aefd515c929e84817681ff598d` (MIT,
`baseline/LICENSE`). It replaces the old task's Composable Kernel oracle, which cannot be vendored as source. The kernel
is reduced to this scenario (unblocked path, no scale upper bound, no clamp or dump) and one change is made: a
round-to-nearest step before the INT8 conversion, which truncates. A persistent grid of `min(tokens, CUs)` programs
handles one row at a time; the CU count comes from `torch.cuda.get_device_properties`.

## Workloads

A token moves 12 KB (8 KB in, 4 KB out), so cases need about 2300 tokens to exceed the 10 us timing floor; performance
cases start at 4096 tokens and span 100x up to 409600 tokens (5 GB moved). The generator and the float64 reference work
in chunks. The old task's 5120 and 8192 widths and one-token cases are not part of this scenario; decode batches are
correctness-only.

## GPU audit risks

The Triton float to INT8 conversion and the `rsqrt` approximation are assumed to truncate and to be accurate to a few
ulp; a baseline code mismatch would show first as a few `exact` differences. The rounding step in the baseline has not
been run on a GPU.
