# Sum over the middle axis, float32, [B, 4096, 4095]

`out[b, 0, m] = sum_n x[b, n, m]` with `keepdim=True`: a strided (non-innermost) reduction over an
unaligned, odd-width contiguous axis, accumulated in float32. The shape family comes from the
KernelBench level-1 "sum reduction over a dimension" problem (reduce dim 1; inputs from `torch.rand`,
here a uniform distribution on [0, 1)), whose cases were 32/64/128 x 2048x2047, 64x4096x4095 and
128x1024x1023. Each (reduce, inner) pair is its own scenario (siblings `f32_d1024_reduce1`,
`f32_d2048_reduce1`, `f32_d4096_reduce1`); this one varies the batch. The original suite's 64x4096x4095 case lies inside this sweep; B=1 (67 MB) is the axis minimum and also a performance case.

Baseline: a standalone Triton rewrite of the reduction kernel torch.compile (Inductor) generates for
`torch.sum(x, dim=1, keepdim=True)`: one program per (64-column block, batch), looping over the
reduced axis in 32-row tiles with a float32 accumulator tile. It is not copied from a specific
Inductor output; provenance is recorded as the PyTorch source it replaces (BSD-3-Clause; the
revision is a nominal release tag).

The reference accumulates in float64 one batch element at a time to keep temporaries small.
The output bound is `baseline_relative` with `k: 3` and `max_violation_fraction: 0.001` (default k is 2, fraction 0).
The baseline sums 32 independent per-row-tile partials of 128 additions each and then tree-reduces them, so
it is considerably more accurate than a plain float32 evaluation of the sum (for example `torch.sum` on float32 inputs):
output values are about 2048, where one float32 ulp is about 1.2e-4. The QA audit's float32 PyTorch evaluation of
the reference failed the default bound in b16, b37, b64 and b128 with violation fractions of 4.4e-5 to 6.1e-5 and a
peak error of 1.05x to 1.15x the bound, with no bias (z below 0.002): a correctly rounded float32-accumulating
candidate with a different (e.g. sequential) summation order lands just outside 2x the baseline's error on a few
elements. `k: 3` puts that peak at about 0.4x to 0.8x of the bound and 0.001 allows a rare outlier (the observed
fraction is at most 6.1e-5), while a shifted or quantized result (the audit's 16-ulp shift and fp8-rounding controls) is
orders of magnitude outside it.
Risk for the GPU audit: the largest input is several GB.
