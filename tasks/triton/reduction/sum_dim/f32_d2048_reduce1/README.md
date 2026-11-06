# Sum over the middle axis, float32, [B, 2048, 2047]

`out[b, 0, m] = sum_n x[b, n, m]` with `keepdim=True`: a strided (non-innermost) reduction over an
unaligned, odd-width contiguous axis, accumulated in float32. The shapes come from the KernelBench
level-1 "sum reduction over a dimension" problem (reduce dim 1; inputs from `torch.rand`, here a
uniform distribution on [0, 1)). The original suite used batches 32, 64 and 128 at 2048x2047; this task fixes that scenario and
varies the batch.
Batches 2 and 256 extend the span to 128x; B=1 is correctness-only (below the 10 us floor). The
other original shapes are the sibling scenarios `f32_d4096_reduce1` and `f32_d1024_reduce1`.

Baseline: a standalone Triton rewrite of the reduction kernel torch.compile (Inductor) generates for
`torch.sum(x, dim=1, keepdim=True)`: one program per (64-column block, batch), looping over the
reduced axis in 32-row tiles with a float32 accumulator tile. It is not copied from a specific
Inductor output; provenance is recorded as the PyTorch source it replaces (BSD-3-Clause).

The reference accumulates in float64 one batch element at a time to keep temporaries small.
The output bound is `baseline_relative` with `k: 3` and `max_violation_fraction: 0.001` (default k is 2, fraction 0).
The baseline sums 32 independent per-row-tile partials of 64 additions each and then tree-reduces them, so it is
considerably more accurate than a plain float32 evaluation (output values are about 1024, one float32 ulp is about
6.1e-5). The QA audit's float32 PyTorch evaluation of the reference failed the default bound in every case with more
than two elements (b1 to b256): violation fractions of 1.1e-4 to 4.9e-4 and a peak error of 1.02x to 1.27x the bound,
with no bias (z at most 0.02). A correctly rounded float32-accumulating candidate with a different (for example
sequential) summation order therefore lands just outside 2x the baseline's error on a few elements. `k: 3` puts that
peak at about 0.7x to 0.85x of the bound, and 0.001 allows a rare outlier (the observed fraction is at most 4.9e-4, a
single element in b1), while a shifted or quantized result (the audit's 16-ulp shift and fp8-rounding controls) is
orders of magnitude outside it and far beyond the 16x peak limit for admitted elements.
Risks for the GPU audit: B=2 has few programs and may be below 10 us; the B=256 input is 4.3 GB.
