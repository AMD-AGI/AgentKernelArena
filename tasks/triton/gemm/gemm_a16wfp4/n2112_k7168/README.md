# a16wfp4 GEMM, N = 2112, K = 7168

`out = mxfp4(x) @ dequant(w)^T`. The activations `x` [m, 7168] arrive in bfloat16 and are quantized to MXFP4
**inside the operator** (that work is part of the timed call): one e8m0 scale per 32 values of K, taken from the
group maximum rounded to a power of two minus the exponent of the largest e2m1 value, then round to nearest even
onto the e2m1 grid, saturating at 6. The weights are MXFP4: `w` [2112, 3584] holds two e2m1 codes per byte (low
nibble is the even K index) and `w_scales` [2112, 224] holds uint8 e8m0 scales (`2^(s - 127)`) per 32 values of K.
The block products are accumulated in float32 and the result is rounded to bfloat16. The shape is a DeepSeek-style
q/kv down projection (2112 = 1536 + 512 + 64 outputs over a 7168-wide hidden state).

**Reference.** `reference.py` reproduces the activation quantization (power-of-two rounding of the group maximum,
value-based round-to-nearest-even onto the e2m1 grid) and multiplies dequantized operands in float64, so quantization
differences between an implementation and the operator are caught as errors rather than hidden in a tolerance.
Row-chunked to bound memory. It was cross-checked against the bit-level emulation of the old FlyDSL task on CPU
(identical dequantized activations, including all-zero and tiny groups).

**Inputs.** `inputs.py` quantizes normal weights (std 0.05) to MXFP4 with e8m0 scales; activations are standard
normal.

**Workloads.** The old tasks (a GEAK sweep of 55 (M, N, K) shapes and a PyTorch-oracle FlyDSL task) both contain
N = 2112, K = 7168; the GEAK sweep used M of 1 to 128 for it and the FlyDSL task used 128 and 256. M is extended
here to prefill token counts so that 100x of var-axis span is reachable: decode-sized counts (1 to 129) are
correctness-only and performance cases run from 512 to 65536 tokens. The baseline's production tile is 8 rows
high, so even M = 512 runs 1088 programs each looping over 14 K steps with an in-kernel quantization, which is
expected to be above 10 us on MI355X although the bandwidth estimate alone is below it.

**Baseline.** AITER's Triton `_gemm_a16wfp4_kernel` with the launch configuration of the gfx950 table (see
`provenance` in `task.yaml`; `baseline/LICENSE` is the AITER MIT license). The previous GEAK copy was taken from a
different AITER revision; this one is from the revision recorded in `task.yaml`. The same table applies to every
M because upstream has no shape-specific table for this N and K (lookup key N = 2112, K = 7168 unpacked) of the
non-preshuffled kernel, so the generic gfx950 table is used.

**GPU audit risks.** The 10 us floor at M = 512 (see above); baseline timing at M = 65536 is long (about 140k small
programs) which lengthens timing rounds; `tl.dot_scaled` needs MI355X (gfx950).
