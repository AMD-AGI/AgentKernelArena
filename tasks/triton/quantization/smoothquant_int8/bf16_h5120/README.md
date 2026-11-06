# SmoothQuant INT8 quantization, bfloat16, hidden size 5120

Applies a per-channel smoothing scale to the activation (`y = x * x_scale`, float32) and then quantizes
each token row dynamically to INT8 with one float32 scale `amax(|y|) / 127` (scale 1 for an all-zero
row). Production origin: AITER `smoothquant_fwd` (a Composable Kernel operator) as used by
SmoothQuant W8A8 INT8 models; the old FlyDSL task covered the same operator.

Semantics worth knowing: the quotient `y / scale` is formed by correctly rounded float32 division,
saturated to `[-128, 127]` and **truncated toward zero** (the device INT8 store), not rounded. Codes
must match bit-for-bit, so the reference works in float32 and reproduces the operator exactly; a
float64 reference, a reciprocal multiply or a rounding conversion would flip codes. This strictness is
intentional (the old task allowed one code of tolerance; v3 requires exact integer outputs). The scales
use the default `baseline_relative` bound. `x_scale` is drawn log-uniformly in `[2^-3, 2^3]` (positive,
spanning magnitudes like real smoothing factors).

Baseline: the production operator is Composable Kernel and cannot be vendored as source, so the
baseline is a Triton rewrite (one program per row), derived from the quantization step of AITER's
Triton `_quant_rms_norm_kernel` with the normalization removed and the division-based truncating
quantization of the task contract. It is a rewrite, not the production kernel. Workloads are prefill
token counts at hidden size 5120; batches below about 2048 tokens run under the 10 us timing floor
and are correctness-only.

Reference correction (round 2): the MI355X re-audit still found a handful of code mismatches per case with
the scales agreeing only to about 146-157 dB SQNR, even though the baseline already used `tl.div_rn`. The
cause was the reference: on a GPU, `torch` evaluates `tensor / python_scalar` as a multiplication by the
float32 reciprocal (`amax * (1/127)`), which differs from the correctly rounded quotient by one ulp for some
rows, and a one-ulp scale difference flips truncated codes. The reference now divides by a same-device
tensor, which uses true IEEE division like the operator and the `tl.div_rn` baseline. The tolerance is
unchanged (exact codes).

Baseline adaptation: the baseline uses `tl.div_rn` for the scale and the quotient, because the ROCm Triton
build lowers `/` to an approximate division and codes are compared exactly. The algorithm is unchanged.

Bound override: `compare.out` is `exact` because the contract requires integer codes to match bit-for-bit.

Inputs: standard-normal rows from the task generator (`inputs.py`) with every 97th row all zero, which
keeps the old task's zero-row controls (row scale 1, zero codes) in the exact comparison.
