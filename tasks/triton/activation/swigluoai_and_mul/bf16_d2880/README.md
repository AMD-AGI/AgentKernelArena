# GPT-OSS SwiGLU and mul, bfloat16, d = 2880

`out = g * sigmoid(1.702 * g) * (u + 1)` with `g = min(gate, 7)` and `u = clamp(up, -7, 7)`; the gate
half comes first and the up half second in each `[n_tokens, 2*d]` row. The constants are fixed by
the operator (AITER's `swiglu_and_mul` takes no limit argument). The reference is float64.

**Origin.** The GPT-OSS expert activation (hidden and intermediate size 2880), as served through
AITER's `aiter.swiglu_and_mul`. The old `torch2flydsl/swiglu_and_mul_kernel` task used d in {2048, 3200,
4096} with three of four shapes under the timing floor; d = 2880 is the real model width.

**Baseline.** The old task's baseline was a PyTorch model; here it is a Triton rewrite (not vendored
code) of AITER's HIP C++ `swiglu_act_and_mul_kernel` (ROCm/aiter at the recorded commit, MIT, license
copied to `baseline/LICENSE`): one program per row and 1024-column block, float32 math, bfloat16
store. The sigmoid uses `1 / (1 + exp(-alpha * g))` instead of the hardware reciprocal.

**Workloads.** Prefill token rows. At 17.3 KB per row about 1600 rows reach 10 us; smaller batches
are correctness-only. Inputs use std 3 so about 2% of gate and up values exceed the limit and the
clamps matter, and the upper tail of the gate saturates.

**Comparison.** Default `baseline_relative`.

**Memory.** The largest case is 3 GB of input; the reference works in 16K-row float64 chunks and
holds a 6 GB float64 output.
