# Dynamic per-tensor FP8 quantization, bfloat16, hidden size 4096

Quantizes a whole `[n_tokens, 4096]` bfloat16 activation with one dynamic scale
`max|x| / 448` to `float8_e4m3fn` (the OCP FP8 type of gfx950) and returns the codes and the
float32 scale of shape `[1]`. The scale is an output, so a candidate must reduce across all rows
before any element can be quantized.

The reference works in float64 and returns the clamped, **unrounded** quotient `x / scale` as the
expected codes (the reference may not return FP8); the default `baseline_relative` bound then
allows one FP8 rounding step plus twice the baseline's own error, and the same bound judges the
scale. Production origin: AITER's `dynamic_per_tensor_quant` (HIP) and its Triton equivalent
`dynamic_per_tensor_quant_fp8_i8`, used for per-tensor FP8 activation quantization; the original
FlyDSL task covered `[1..4096, 4096..8192]` bf16 inputs.

Workloads: token counts from decode batches (correctness only) to batched prefill. Cases below
about 2048 tokens run faster than 10 us (three launches, little data) and are correctness-only;
the timed cases span 2048 to 262144 tokens.

Baseline: the AITER Triton kernels (amax with `atomic_max`, then quantize), vendored with the
modifications listed in `task.yaml`. The task's image (HIP) kernel is not used as the baseline
because it needs AITER headers and rocprim. Floating atomics make the amax order-independent, so
the scale is deterministic.

Scale bound: the scale is compared with `tolerance` (`rtol` 2.5e-7, `atol` 0) rather than
`baseline_relative`. The baseline's float32 scale equals the float64 reference rounded to float32,
so its own error is at most half a float32 ulp and the default bound would be about one ulp. The QA
calibration probe showed a float32 PyTorch evaluation of the reference (for instance `amax * (1/448)`
instead of `amax / 448`, a common and correct implementation) lands one ulp away and fails it by 1.39x
at `t3`. A float32 ulp is at most 2^-23 (1.19e-7) relative, so `rtol` 2.5e-7 admits two ulps. Wrong
scales are still rejected: a different divisor (447 or 449) is off by 2e-3, a bf16-rounded scale by up
to 4e-3, and a scale from a partial amax is off by far more. The `qx` bound is unchanged.
