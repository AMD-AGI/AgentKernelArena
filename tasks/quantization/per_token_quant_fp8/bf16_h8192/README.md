# Dynamic per-token FP8 quantization, bfloat16, hidden size 8192

Quantizes each row of a `[n_tokens, 8192]` bfloat16 activation with its own dynamic scale
`max|x| / 448` to `float8_e4m3fn` (the OCP FP8 type of gfx950) and returns the codes and the float32
per-token scales of shape `[n_tokens, 1]` (the layout of AITER's per-token quant op). The scale is an
output, so each row needs its amax before any element can be quantized. Hidden sizes of 4096 to 8192
appear in the old FlyDSL tasks; 8192 is the widest. Per-tensor and static modes have their own task
(`per_tensor_quant_fp8`); the int8 mode of the old suite is not part of this scenario.

The reference works in float64 and returns the **unrounded** quotient `x / scale` as the expected
codes (a reference may not return FP8) plus the float64 scales. The default `baseline_relative`
bound allows one FP8 rounding step plus twice the baseline's own error, and judges the scales the
same way. Codes are therefore not compared bit-for-bit: the production kernel multiplies by the
float32 reciprocal of the scale, which differs from an exactly rounded division by one float32 ulp
and, at the midpoint between two FP8 codes (for example a normalized value of exactly -76 in a row
whose scale is 3/512), selects the neighbouring code. That one-step difference is within the bound;
an exact comparison would reject the production operator. All-zero rows (scale 0) are not generated.

Baseline: AITER's Triton `dynamic_per_token_quant_fp8_i8` (one program per row, whole row in
registers), vendored from ROCm/aiter with the only change of removing `tl.assume(pid > 0)`, which
program 0 violates. Workloads: token counts from decode to batched prefill; cases below about 1280
tokens may run under the 10 us timing floor and are correctness-only. Performance cases span
1280 to 131072 tokens (102x).

The workload set stops at 131072 tokens (2 GiB of bfloat16 input). An earlier 262144-token
case passed the baseline's correctness check but ran the candidate out of memory on the 288 GB
MI355X audit (a 32 GiB allocation failed with 256 GiB already held): the float64 reference codes
alone are 16 GiB and the comparison holds several full-size float64 copies of them. That case also
reaches 2**31 elements, where the baseline's 32-bit element offsets overflow. The 131072-token case
is the same bandwidth-bound regime and adds no coverage beyond memory pressure.

Scale bound: the scale is compared with `tolerance` (`rtol` 7.5e-7, `atol` 0, as for the `y_s` scales of
`moe/silu_and_mul_fp8_quant_masked`; the comparator adds the float32 unit roundoff u = 2^-24 = 5.96e-8, so
the per-element limit is 8.1e-7 = 13.6 u of the reference scale) instead of `baseline_relative`. The baseline's float32 scale is the float64 reference rounded to float32, so the
default bound is tight, and the QA calibration probe showed a float32 PyTorch evaluation of the
reference failing only the bias test on the scale in every case (violation fraction 0, peak about
0.8x of the old bound, bias z 0.83-0.85): the float32 rounding of `amax / 448` is a systematic offset
shared by all rows, so no `k` or violation fraction can absorb it. A float32 CPU emulation over 8
draws of 16384 rows each measured a worst relative scale error of 5.0e-8 for `amax / 448` and
1.02e-7 for `amax * (1/448)` (the common reciprocal form); over every bfloat16 `amax` mantissa these
are at most 0.85 u and 1.71 u. The baseline divides with ROCm Triton's approximate `/`, whose error on
`amax / 448` was not measured on the GPU. The limit leaves room for it: even a scale 2 float32 ulps
(at most 4.8 u) from the exact quotient uses only 35% of the limit, so a correct float32 candidate,
the baseline and its copy pass with at least 2.8x margin on the worst element. Wrong scales are still
rejected: the `shifted_16_ulp` control moves the scale by 16 u, which after float32 rounding is 14.5 u
to 18.5 u from the reference over every bfloat16 `amax` mantissa, so it exceeds the 13.6 u limit on
every scale (on 31% of them even if the baseline were 2 ulps low). It also doubles every FP8 code,
which `qx` rejects on its own. The `fp8_rounded` control rounds the scale to FP8 (relative error up to
6%), a different divisor (447 or 449) is off by 2e-3, and a partial amax is off by far more. The other controls (noop, zeros,
half_written, stale_outputs, input_write) change the scale or the codes by far more than the bound,
and `qx` keeps the unchanged `baseline_relative` bound, which already rejected all of them in the
r3 audit.
