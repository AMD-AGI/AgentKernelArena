# Per-token-group FP8 quantization with column-major scales, bfloat16, hidden size 7168, group 128

The same operator as `../bf16_h7168_g128`: each row of a `[n_tokens, 7168]` bfloat16 activation is
split into 56 groups of 128 values, each quantized to `float8_e4m3fn` with scale
`max(max|x|, 1e-10) / 448`. Here the float32 scales are written **column-major**, the layout that
DeepGEMM-style and AITER blockscale GEMMs read (vLLM's `column_major_scales=True`).

Because an output cannot carry a layout transform, the scale output is declared as the contiguous
`[num_groups, n_tokens]` tensor `x_s`, with `x_s[g, t]` the scale of group `g` of token `t`. This is
byte-for-byte the storage of the `[n_tokens, 56]` scale matrix with strides `(1, n_tokens)`; the
reference returns the transposed scales accordingly. Scattered writes of one scale per group into
56 planes are what distinguishes this task from the row-major scenario.

The reference works in float64 and returns the clamped, **unrounded** `x / scale` as the expected
codes (a reference may not return FP8) plus the float64 scales; the default `baseline_relative`
bound allows one FP8 rounding step plus twice the baseline's own error, and judges the scales the
same way (per-row RMS over the token axis). No bound is overridden.

Baseline: vLLM's Triton `_per_token_group_quant_fp8_colmajor` (one program per group), vendored with
its `per_token_group_quant_fp8` launch (the Triton fallback; on ROCm vLLM prefers a native kernel
for contiguous inputs). The old task shipped the same kernel with a division by the FP8 maximum;
this version uses the upstream multiply-by-reciprocal form. Group size, epsilon and
`use_ue8m0=False` are fixed by the scenario.

Workloads: token counts from decode to batched prefill.
Cases below about 2048 tokens may run under the 10 us timing floor and are correctness-only.

The old task also ran a group size of 64 on small shapes; the group size is a scenario constant here
(128, the production value for blockwise W8A8 GEMMs), so group 64 is not covered.
