# LayerNorm over a (64, 256, 256) volume, float32

Normalizes each sample over all 64 x 256 x 256 = 4194304 of its elements (mean and biased variance of the
whole sample), then applies an elementwise `weight` and `bias` of the same shape. This is the KernelBench
level-1 LayerNorm problem. The reference computes in float64, sample by sample. Inputs are uniform in [0, 1)
as in the KernelBench generator (a large mean relative to the spread, which punishes a naive sum-of-squares
variance); weight and bias use trained-parameter-like distributions instead of the previous task's
index-dependent ramps, which only served to make both affine terms non-trivial.

Baseline: the previous task's baseline was `torch.nn.LayerNorm`, which cannot be vendored. It is replaced by
a Triton rewrite of the split reduction TorchInductor generates for such a row (see the provenance in
`task.yaml`; this is not a verbatim Inductor dump): kernel 1 computes per-chunk (8192 elements) mean and
squared deviations, kernel 2 merges the chunks per sample, kernel 3 normalizes and applies the affine
transform. Scratch buffers for the partial statistics are allocated inside `run()`. Statistics are
accumulated in float32 from chunk-local deviations.

Workloads: batches 1 to 128 (128x). The sample is 16 MiB (float32), so even batch 1 moves well over 50 MiB
and is far above the 10 us floor. Batches 3, 20 and 50 are correctness-only boundary cases. The largest
case holds 2 GiB of input and output each; the float64 reference is computed per sample to bound memory.

## Error bound

`out` uses `sqnr` with `min_db: 128`. The error of a correct float32 LayerNorm is dominated by the float32 rounding of
the per-sample mean and reciprocal standard deviation, whose size depends on the summation order over 4194304 terms.
Every element of a sample shares that shift, so the elementwise error is correlated, not independent noise: the
`baseline_relative` bound (zero violations, k=3) rejected a float32 evaluation of the reference in 10 of 11 cases of the
MI355X audit (violation fraction up to 0.068, peak 4.0x), and so could a correct float32 candidate. The baseline's
chunked statistics are more accurate than a plain float32 accumulation, so the bound cannot be anchored to its error.

`sqnr` compares the error norm of the whole output with the norm of the reference (the reference rounded to float32 is
the signal). CPU emulation (3 samples of the task's distributions, one per run) gives, for correct float32 designs:
torch two-pass float32 138.6 to 144.0 dB, sequential float32 sums 138.6 to 145.4 dB, 1024-wide blocked sums 136.7 to
145.4 dB, and the worst case, a one-pass `E[x^2] - mean^2` with sequential sums, 134.0 to 140.0 dB. The control
`shifted_16_ulp` (output scaled by 1 + 16 u) is 120.4 dB, 7.6 dB below the bound; a 4-ulp shift is 132.5 dB and an 8-ulp
shift is about 126.5 dB, so a row-wise shift of roughly 6 output ulps or more is rejected. The threshold leaves 6 dB (2x
in amplitude) under the least accurate correct float32 design. Gross errors are far outside it: `fp8_rounded` is 31.5 dB,
a skipped or zeroed 1% of the output 20 dB; the zeros, stale and half-written controls are orders of magnitude lower.
A lone wrong element of small magnitude is not detected by a norm-based bound.
