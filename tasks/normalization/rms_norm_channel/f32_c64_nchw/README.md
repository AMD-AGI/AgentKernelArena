# RMS normalization across channels, float32 NCHW, 64 channels

`out = x / sqrt(mean_c(x^2) + epsilon)` over the channel axis of a `[batch, 64, height, width]` float32 tensor, with
epsilon 1e-5 and no learnable weight. This is the KernelBench level-1 `RMSNorm_` problem. It differs from the language
model `rms_norm` tasks: the reduced axis is the strided channel axis (stride `height * width`), not the contiguous
last axis, so every pixel is normalized across 64 values that lie `height * width` elements apart. Inputs are
uniform in [0, 1) as in the KernelBench generator. The reference computes in float64, sample by sample.

## Baseline

KernelBench's baseline is the eager PyTorch module (`x / sqrt(mean(x ** 2, dim=1, keepdim=True) + eps)`), which cannot
be vendored. It is replaced by a Triton rewrite of the fused reduction-plus-pointwise kernel that TorchInductor
generates for this expression (provenance in `task.yaml`; not a verbatim Inductor dump): each program takes 128 pixels
of one sample, loads the `[64, 128]` channel tile once (64 contiguous runs of 512 B), reduces over channels in float32
and writes the normalized tile, so every element is read and written once. The kernel needs no scratch buffers.
The channel count is a compile-time constant (64).

## Workloads

The image-size and batch combinations are those of the KernelBench problem: batches 16, 32, 64, 112 and 32 at 128 x
128, 128 x 128, 256 x 256, 256 x 256 and 512 x 512, plus a batch-4 case at 128 x 128 to widen the range. Each pixel
moves 512 B, so a batch-4 128 x 128 case moves 33 MB; performance cases span 128x in pixels (65536 to 8388608; the
largest moves 4.3 GB). A single pixel, odd batches and unaligned image sizes (17 x 19, 100 x 77, 129 x 131) are
correctness-only boundary cases. The float64 reference is computed one sample at a time to bound memory.

## Error bound

`out` uses `baseline_relative` with `k: 4` and `max_violation_fraction: 0.001` (default: k=2, no violations). Each
output element is `x / sqrt(m + epsilon)` with `m` the float32 channel mean of squares; the error of an implementation
is a few float32 roundings (the 64-term sum, the square root and the division), and the baseline's error is itself only
about one rounding, so the default bound `2 x max(|baseline error|, row RMS) + one output rounding` is only a couple of
units of roundoff wide. A float32 PyTorch evaluation of the reference (a different summation order and an exactly
rounded division) therefore exceeded it for 0.1 to 0.2% of elements in every case of the MI355X audit (peak 1.5x), and so
could a correct float32 kernel. CPU emulation (batches of 2 to 4 at 17 x 19 up to 256 x 256, the task's uniform draws,
a baseline emulated with a pairwise float32 channel sum and either an exact division or a multiplication by the
reciprocal, the float32 PyTorch evaluation as the candidate): at k=2 the violation fraction is 1.0e-3 to 2.2e-3 (peak up
to 1.7x); at k=3 it is up to 5.6e-4; at k=4 it is up to 1.6e-4 (peak up to 1.3x, far below the 16x peak limit)
with the bias test passing in all of them (z about 0.3, accepted against the baseline's own bias). k=4 with a
0.1% allowance keeps a margin of about 6x over the worst emulated fraction. The bound still rejects the known-wrong
controls: `shifted_16_ulp` is a 16-roundoff relative error, about five times the whole bound, and stale, zeroed
or half-written outputs miss it by orders of magnitude.
