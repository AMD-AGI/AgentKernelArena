# gelu_tanh_and_mul, bfloat16, d = 4096

Operator: `out = gelu_tanh(x[:, :d]) * x[:, d:]` with the tanh-approximate GELU
(`0.5 g (1 + tanh(sqrt(2/pi)(g + 0.044715 g^3)))`), the gated activation in Gemma-style MLPs. Input
rows hold `2 * d = 8192` bfloat16 values, the output `d = 4096`. Production origin: AITER's
`gelu_tanh_and_mul` runtime op (the former `torch2flydsl/gelu_tanh_and_mul_kernel` task).

Scenario: bfloat16, `d = 4096` fixed, `n_tokens` varies (1 to 131072).

Workloads: prefill chunks and batched prefill from 1280 to 131072 tokens (102x span; the smallest
moves 31 MB, estimated at about 11 us). Decode-sized batches (1, 3, 48) and an unaligned 1000 tokens
are correctness-only because they run under the 10 us timing floor. Inputs are normal with std 2.

Reference: float64 tanh GELU times the up half, in row chunks. The default `baseline_relative`
bound is used. The Triton `tanh` in the baseline is `2 * sigmoid(2x) - 1`, which loses relative
precision where tanh is near zero; this affects only tiny outputs and is covered by the bound.

Baseline: a Triton kernel derived from AITER (MIT, `ROCm/aiter@652f2d4b`). AITER ships no plain
GELU-tanh-and-mul Triton kernel, only `fused_silu_mul_kernel` and quantizing variants; the baseline
is the `fused_silu_mul_kernel` tile structure and launch heuristics with the unchanged AITER
`_gelu_tanh` device function, the product taken in float32 and rounded once. `baseline/LICENSE` is
the AITER MIT license. This is a modification, not byte-for-byte vendoring.

GPU risks: not yet run. The largest case holds a 4.3 GB float64 reference output plus comparison temporaries (estimated well under 40 GB).
