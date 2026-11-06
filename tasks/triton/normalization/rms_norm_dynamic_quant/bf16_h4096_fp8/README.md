# RMSNorm with dynamic per-token FP8 quantization, bfloat16 in, hidden size 4096

Each bfloat16 token row is normalized, `y = x * rsqrt(mean(x^2) + epsilon) * weight` (epsilon 1e-5), without rounding
`y` to bfloat16, then quantized with a per-token scale: `scale = absmax(y) / 448` (1 for an all-zero row) and the
codes are `y / scale` in `float8_e4m3fn` (OCP format, maximum 448, gfx950). The scales are float32 of shape
`[n_tokens, 1]`. This is the activation quantization in front of an FP8 GEMM (`rmsnorm2d_fwd_with_dynamicquant`).

The planned scenario name said int8; the source task (`torch2flydsl/rmsnorm2d_dynamicquant_kernel`) quantizes to FP8
(`float8_e4m3fn` on gfx950, `float8_e4m3fnuz` on gfx942), so this scenario is FP8 and the directory is named
accordingly. The old oracle was the CK operator `aiter.rmsnorm_quant`; the old harness accepted codes within one
byte and scales within 1e-3 of the maximum scale.

`weight` is standard normal, as in the old task's harness, so it has both signs and the absolute maximum is exercised
with negative products.

The reference is float64 end to end, in token chunks, and returns the unrounded `y / scale` as the codes and the
float64 scale. Codes and scales are floats, so both outputs use the default baseline-relative bound: a code may
differ from the reference by twice the baseline's own error (its FP8 quantization error) plus the e4m3 rounding, and
the baseline must reach 25 dB SQNR (the FP8 format gives about 32 dB). The scale is a single float32 per token, so a
candidate computing it as `absmax / 448` in float32 agrees with the baseline to float32 rounding.

Scale bound: the scale is compared with `tolerance` (`atol` 0, `rtol` 6e-7; the limit is `rtol + 2^-24` of the
reference scale, 6.6e-7) instead of `baseline_relative`. The scale is one float32 per token, and the baseline's scale is
the float64 reference rounded to float32 plus the rounding of its own float32 sum of squares, about one float32 ulp. A
correct candidate that accumulates the sum of squares in float32 in another order, or computes `absmax * (1/448)`, differs
from the reference by a few ulp, and its error is correlated within the case: a float32 PyTorch evaluation of the
reference failed the `baseline_relative` bound (`k` 4) on the bias test in every case from 1000 tokens up (bias z about
0.62, violation fraction 0 to 3.1e-5, peak 0.75x to 1.26x of that bound), so no `k` could accept it. A tolerance bound waives
the bias test when no element violates.

Calibration: a CPU emulation of the float32 scale path over 16 draws of 16384 tokens (262144 tokens) gives a worst
relative error of 2.6e-7 for `absmax / 448` after a float32 `rsqrt(mean(x^2) + eps)`, 2.7e-7 for a reciprocal-square-root
scale applied as `(x * r) * w`, and 3.0e-7 for `absmax * (1/448)` (which also carries a systematic +4.5e-8 mean offset);
the largest error in the MI355X probe is about 3e-7 (peak 1.26x of the old bound, which was about 2.3e-7). The new limit
of 6.6e-7 is at least 2.2x the worst of these, so every probe row (t1 to t262144) passes with no violating element.
Wrong scales stay far outside: a divisor of 447 or 449 is off by 2.2e-3, a bfloat16-rounded scale by up to 2e-3 to 4e-3,
a scale from a partial absmax by far more, and the `shifted_16_ulp` control multiplies the scale by `1 + 16 * 2^-24`
(9.5e-7, 8.9e-7 after the float32 rounding), 1.35x the limit, so every element of `scale` is rejected under it (the
control is also rejected through `out`, whose fp8 codes it doubles). `fp8_rounded` rounds the scale to float8_e4m3fn
(relative error up to 6%) and is rejected on `scale` and `out`. The other controls (noop, zeros, half_written,
stale_outputs, input_write) change the output or the input far beyond a few ulp. The `out` bound is unchanged.

## Baseline

AITER's Triton `_quant_rms_norm_kernel` and `_per_token_quant` (`aiter/ops/triton/_triton_kernels/normalization/
rmsnorm.py`, commit `652f2d4b768239aefd515c929e84817681ff598d`), unchanged: a persistent kernel with one program per
compute unit (`NUM_PRGMS = min(n_tokens, CU count)`, 4 warps), one 4096-wide block per row, which computes the
sum of squares, the normalized row in float32, its absolute maximum, the scale (stored as `absmax / 448`) and the
codes (`y * (1 / scale)`) in registers. The upstream launcher is reduced to this scenario in `run()`; the smooth-scale,
scale-upper-bound, clamp, intermediate-dump and blocked (hidden size above 32768) variants of the kernel stay in the
file but are not exercised.

## Workloads

A token moves about 12 KB (8 KB bfloat16 in, 4 KB FP8 out). The MI355X audit measured the baseline at 24 us for 4096
tokens, 64 us for 16384 and 0.90 ms for 262144 (about 3.2 us per 1000 tokens plus a fixed cost of about 11 us), so the
2048-token case of the old task is estimated at about 17 us and is a performance case; decode batches (1, 3, 8, 48)
and the unaligned batch of 1000 are under the 10 us floor and are correctness-only. Performance cases span 128x, from
2048 to 262144 tokens (3.2 GB moved).

The workload set stops at 262144 tokens. An earlier 524288-token case passed the baseline's correctness check but ran
the candidate out of memory on the 288 GB MI355X audit (a 32 GiB allocation failed with 256 GiB already held): the
float64 reference outputs alone are 16 GiB and the comparison holds several full-size float64 copies of the codes. The
case measured 1.75 ms, the same bandwidth-bound regime as 262144 tokens, so it added no coverage beyond memory pressure.
