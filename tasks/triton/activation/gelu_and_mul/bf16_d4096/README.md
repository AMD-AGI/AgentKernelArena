# gelu_and_mul, bfloat16, d = 4096

Operator: `out = gelu(x[:, :d]) * x[:, d:]` with the exact (erf) GELU, the gated activation after a
fused gate/up projection in GELU-gated MLPs. Input rows hold `2 * d = 8192` bfloat16 values, the
output `d = 4096`. Production origin: AITER's `gelu_and_mul` runtime op (the former
`torch2flydsl/gelu_and_mul_kernel` task, which compared against it).

Scenario: bfloat16, `d = 4096` fixed, `n_tokens` varies (1 to 131072). The old task also used
input widths 4096, 6400 and 8192 (d = 2048, 3200, 4096) at m = 1..4096; this task keeps d = 4096.

Workloads: prefill chunks and batched prefill from 1280 to 131072 tokens (the performance cases
span 102x; the smallest moves 31 MB, estimated at about 11 us). Decode-sized batches (1, 3, 48) and an
unaligned 1000 tokens are correctness-only because they run under the 10 us timing floor. Inputs
are normal with std 2 so the GELU saturation region is exercised.

Reference: float64 erf GELU times the up half, computed in row chunks to bound memory. The default
`baseline_relative` bound is used.

Baseline: a Triton kernel derived from AITER (MIT, `ROCm/aiter@652f2d4b`). AITER ships no plain
GELU-and-mul Triton kernel, only `fused_silu_mul_kernel` and quantizing variants; the baseline
is the `fused_silu_mul_kernel` tile structure and launch heuristics with the unchanged AITER
`_gelu` device function, the product taken in float32 and rounded once to match the HIP operator.
`baseline/LICENSE` is the AITER MIT license. This is a modification, not byte-for-byte vendoring.

GPU risks: not yet run. The largest case holds a 4.3 GB float64 reference output plus comparison temporaries (estimated well under 40 GB).
