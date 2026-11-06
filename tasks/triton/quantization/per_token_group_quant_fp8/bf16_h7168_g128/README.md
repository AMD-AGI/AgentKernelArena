# Per-token-group FP8 quantization, bfloat16, hidden size 7168, group 128

Splits each token row into 56 groups of 128 values and quantizes each group to `float8_e4m3fn`
with its own scale `max(max|x|, 1e-10) / 448`; scales are float32, row-major `[n_tokens, 56]`.
This is the DeepSeek-style 1x128 activation quantization feeding blockwise-FP8 GEMMs (hidden size
7168); the old FlyDSL `per_1x128` task is the same operator with group 128.

The reference works in float64 and returns the clamped, **unrounded** `x / scale` as the expected
codes (a reference may not return FP8) plus the float64 scales; the default `baseline_relative`
bound allows one FP8 rounding step plus twice the baseline's own error, and judges the scales the
same way.

Baseline: vLLM's Triton `_per_token_group_quant_fp8` (one program per group), vendored from the
Triton fallback path; on ROCm vLLM prefers a native kernel for contiguous inputs. The UE8M0 and
column-major scale variants are not part of this scenario. Cases below about 1024 tokens may
run under the 10 us timing floor and are correctness-only.

Inputs: `inputs.py` draws every group as a standard normal times 2^e (e uniform in [-12, 6]) and
makes about one group in 32 all zero. This keeps the dynamic-range and all-zero-group (eps floor)
controls of the original vLLM task in every case; the input dtype is fixed to bfloat16 and the
group size to 128 by the scenario (the original task also swept float16/float32 inputs and
other group sizes).
