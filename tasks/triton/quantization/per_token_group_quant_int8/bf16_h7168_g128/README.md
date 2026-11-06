# Per-token-group INT8 quantization, bfloat16, hidden size 7168, group 128

Splits each token row into 56 groups of 128 values and quantizes each group to `int8` with its own
float32 scale `max(max|x|, 1e-10) / 127` (row-major `[n_tokens, 56]`). This is the W8A8 blockwise
INT8 activation quantization of vLLM and SGLang (`per_token_group_quant_int8`).

Semantics worth knowing: the quotient `x / scale` is clamped to `[-128, 127]` and then **truncated
toward zero** (the Triton float-to-int conversion), not rounded. The old task compared codes with an
absolute tolerance of one; the contract requires integer outputs to match exactly, so the reference
reproduces the operator bit-for-bit: it works in float32, forming the scale and the quotient with
IEEE correctly-rounded division. A float64 reference would disagree on elements whose float32
quotient is an exact integer (for example the group maximum, which quantizes to 127 only if
`absmax / (absmax / 127)` rounds to at least 127 in float32). An implementation that replaces the
division by a reciprocal multiply can therefore differ in a few codes and fails; this strictness is
intentional (`compare.x_q` is therefore `exact`, the one departure from the default bound). The scales are judged with the default `baseline_relative` bound.

Baseline: vLLM's Triton `_per_token_group_quant_int8` (one program per group), vendored from the
Triton fallback path; on ROCm vLLM prefers a native kernel. Workloads are the token counts of the
FP8 scenario at the same hidden size; cases below about 2048 tokens may run under the 10 us timing
floor and are correctness-only.

The old task also ran a group size of 64 on small shapes; the group size is a scenario constant here
(128, the production value for blockwise W8A8 GEMMs), so group 64 is not covered.

Baseline adaptation (QA on MI355X, Triton 3.8.0 ROCm): the vendored kernel's `/` on float32 produced
scales about one ulp off (baseline_sanity SQNR about 152 dB instead of exact) and 0.03% of the codes
differing from the IEEE float32 reference, i.e. the ROCm build lowers `/` to an approximate division.
The reference was right (the operator is defined with correctly-rounded float32 division); the baseline
now uses `tl.div_rn` for the scale and the quotient. The algorithm is unchanged.

Reference correction (round 2): the MI355X re-audit still found about 3e-4 of the codes mismatching and the
scales agreeing only to about 152-155 dB SQNR, even though the baseline already used `tl.div_rn`. The cause
was the reference: on a GPU, `torch` evaluates `tensor / python_scalar` as a multiplication by the float32
reciprocal (`absmax * (1/127)`), which differs from the correctly rounded quotient by one ulp for some
groups, and a one-ulp scale difference flips truncated codes. The reference now divides by a same-device
tensor, which uses true IEEE division like the operator and the `tl.div_rn` baseline. The bounds are unchanged
(exact codes). Not re-run on a GPU here; the same correction resolved the identical symptom in
`smoothquant_int8/bf16_h5120`.
