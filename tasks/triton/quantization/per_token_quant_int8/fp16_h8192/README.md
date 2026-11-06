# Per-token INT8 quantization, float16, hidden size 8192

Dynamic symmetric INT8 quantization of an activation with one float32 scale per token row, as used
for W8A8 INT8 GEMMs (vLLM `per_token_quant_int8`, also used on the SGLang INT8 path).

Semantics worth knowing: the scale is `max(max|x|, 1e-10) / 127`, but the codes are formed with the
**reciprocal** `127 / max(max|x|, 1e-10)` and rounded to the nearest integer with ties **away from
zero** (the ROCm `round`), not half-to-even and not truncated. Codes must match bit-for-bit, so the
reference reproduces the operator in IEEE float32 (the float16 inputs and the absolute maximum are
exact in float32); a float64 reference or a different rounding rule would disagree on ties and
near-ties. This strictness is intentional. The scales use the default `baseline_relative` bound.
The old task allowed one code of tolerance; the v3 contract requires integer outputs to be exact.
An all-zero row is clamped by the 1e-10 floor (scale 1e-10/127, zero codes).

Baseline: vLLM's Triton `_per_token_quant_int8` (one program per row), wrapped in `run()`. The old
FlyDSL per-token INT8 task (AITER HIP semantics: scale `amax/127`, truncating conversion, zero row
scale 1) is a different operator contract and is not covered here. Workloads are the token counts of
the sibling normalization/quantization tasks at hidden size 8192; batches below about 2048 tokens
run under the 10 us timing floor and are correctness-only.

Bound override: `compare.out` is `exact` because the contract requires integer codes to match bit-for-bit.

Inputs: standard-normal rows from the task generator (`inputs.py`), plus every 97th row all zero and
every 97th row an integer row in [-254, 254] with maximum 254, which makes the code multiplier exactly
0.5 so odd integers hit exact rounding ties (round-half-away vs half-even vs truncation) and keeps the
old task's zero-data coverage. This is the float16 sibling of `bf16_h8192`; the old Triton task used float16 input.
