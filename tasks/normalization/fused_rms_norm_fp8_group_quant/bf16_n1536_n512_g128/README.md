# Fused RMSNorm and FP8 group quantization, two bfloat16 inputs (1536 and 512 features)

Two RMSNorms in one launch, as at the start of DeepSeek V2/V3 multi-head latent attention: the compressed
query `x1` (`q_a`, 1536 features) is normalized and quantized to `float8_e4m3fn` in groups of 128 columns for the
following block-scaled FP8 GEMM, and the compressed key/value latent `x2` (`kv_a`, 512 features) is normalized and
stays bfloat16. `epsilon` is 1e-6 for both. `scale = max(absmax(y), 1e-10) / 448` per (token, group) and the codes
are `y / scale` clamped to +-448; the normalized `y` is not rounded to bfloat16 before quantization, as in the
production kernel. The scales are row-major `[n_tokens, 12]`. The original task used the MI300 `float8_e4m3fnuz`
format; this task uses the OCP `e4m3fn` format of gfx950 (maximum 448).

The reference is float64 end to end, with no intermediate rounding, and works in token chunks. Codes and `out2` are
floats and use the default baseline-relative bound; the code comparison also requires the baseline
to reach 25 dB SQNR (the FP8 format gives about 32 dB).

Scale bound: `out1_scale` uses `tolerance` (`atol` 0, `rtol` 6e-7) instead of `baseline_relative`. The scale is
`absmax(y) / 448` where `y` comes from a float32 sum of 1536 squares, an `rsqrt` and two products, so a correct
float32 implementation differs from the float64 reference by several float32 ulps, while the baseline's own error
makes the default bound about one ulp wide. The QA calibration probe showed a float32 PyTorch evaluation of the
reference failing that bound (violation fraction 0.0139, peak 1.86x at `t48`). A CPU emulation of float32
evaluations (sequential-order or tree sums, `amax / 448` or `amax * (1/448)`; 160000 random rows, same input
distributions) gave a maximum relative scale error of 2.9e-7 to 3.6e-7 (2.4 to 3.0 ulp), so `rtol` 6e-7 (plus the
float32 unit roundoff 6e-8 the comparator adds) accepts correct float32 designs with margin for the largest case.
Wrong scales are still rejected: a divisor of 447 or 449 is off by 2e-3, a bfloat16-rounded scale by up to 4e-3, and
the 16-ulp shift control (relative 9.5e-7) exceeds the limit of 6.6e-7. The code and `out2` bounds are unchanged.

## Baseline

AITER's Triton `_fused_rms_fp8_group_quant_kernel` (`aiter/ops/triton/_triton_kernels/quant/fused_fp8_quant.py`) at
commit `652f2d4b768239aefd515c929e84817681ff598d`, with its `_rmsmorm_op` and `_fp8_quant_op` helpers, unchanged:
one program per token, a 2048-wide block covering both rows, 4 warps (the upstream heuristic). The upstream
launcher `fused_rms_fp8_group_quant` is reduced to this scenario in `run()`. The old task's other variants
(residual add, flatten, reduce, per-tensor static quantization) and its vendored older copy of the kernel are not part
of this scenario. The upstream file carries no per-file header; its license (MIT, AMD) is in `baseline/LICENSE`.

## Workloads

A token moves about 6.7 KB (3 KB + 1 KB in, 1.5 KB + 48 B + 1 KB out), so cases need about 6k tokens to exceed the 10 us
timing floor: performance cases start at 8192 tokens and span 104x up to 851968 tokens (5.7 GB moved). The largest
case needs about 14 GB for the float64 reference. Decode batches and short chunks are correctness-only.
