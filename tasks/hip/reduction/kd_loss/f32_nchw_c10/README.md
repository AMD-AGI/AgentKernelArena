# Knowledge-distillation loss, float32, NCHW with 10 classes at 64x64

`pixel_kl[n, h, w] = sum_c target * (log(target) - log_softmax(input / T, dim=1))` and
`out = sum(pixel_kl) * T^2 / n`: the KL divergence of a soft target distribution from the
temperature-softened prediction at every pixel, and the distillation loss
(`nn.KLDivLoss(reduction="sum")` scaled by `T^2` and divided by the batch size `n`) as one float32
scalar. `0 * log(0)` counts as 0. The softmax runs over the channel axis (dim 1) of
`[n, 10, 64, 64]` tensors; the task fixes the temperature at 4 and varies the batch `n`.

Origin: the hip2hip `KDLoss` task (image-based distillation). Its original cases were
`[8|16, 10, 32|64, 32|64]`; this task fixes 10 channels at 64x64 and sweeps the batch from 16 to
2048 images for the performance cases (128x span; the baseline takes 0.80 ms at n=16 and 101 ms at
n=2048 on MI355X). The single image and the odd and unaligned batches 3, 17 and 100 are
correctness-only.

Inputs: the student logits (`input`) are normal with standard deviation 8, and the targets come
from the task generator (`inputs.py`) as `softmax(teacher / T)` over the channel axis for teacher
logits of the same spread, so each pixel's targets sum to 1 and both distributions are softmaxes of
N(0, 2^2) values after the temperature, as for two confident networks. The per-pixel KL then has
mean 2.4. The first version drew the student logits with standard deviation 1 and the teacher's
after the temperature with standard deviation 1: the student was nearly uniform after `T = 4`, the per-pixel KL was only about 0.4 next to log-probability terms
near `log(10) = 2.3`, and the float32 rounding of those terms (they cancel to the small KL) was as
large as a 16-ulp error of the result, so no bound could tell the two apart (below).

Why the per-pixel KL is an output: the original operator returns only the scalar. One number cannot
show that every pixel was computed, and a 16-ulp error on it (the QA auditor's `shifted_16_ulp`
control, which passed the first version) is smaller than the legitimate differences of float32
implementations of a sum of 8.4 million terms. The per-pixel map is what
`F.kl_div(log_softmax(input / T, 1), target, reduction="none").sum(1)` gives and what pixel-wise
distillation of dense predictions reduces; masking ignored pixels or weighting pixels needs it
per pixel. The baseline already computed each pixel's value in its block reduction; storing it
costs 4 bytes per 80 bytes read. Every pixel is now checked, and the scalar checks the reduction
and the `T^2 / n` scaling.

Baseline: the provided HIP kernel of the original task, one block per (image, row, column) slice
that computes the max, the sum of exponentials and the KL contribution with block reductions,
stores the slice's KL into `pixel_kl` and adds it into a scalar with `atomicAdd`; a one-thread
kernel then applies `T^2 / n`. Changes: the pybind module is replaced by the `run()` torch op, no
`.contiguous()` copies are made, the block's KL is also stored, and the running sum is a float64
scratch scalar (`at::zeros` on the current stream, so capturable) with the final `T^2 / n` scaling
and float32 cast written into the caller's output. The original accumulated into the float32
output with float32 atomics; the GPU audit showed that this is wrong at large batch: at n=2048
(first inputs) the running sum reaches 3.4e6, where the float32 spacing is 0.25, while each slice
adds about 0.4, so the rounding of every addition left the result 1.2e-3 relative off the float64
reference (n=1024 was 8e-5). The accumulator precision is the only algorithmic change.

Bounds (both depart from the default `baseline_relative`):

- `pixel_kl`: `sqnr` with `min_db: 128`. The error of a correct float32 pixel is a few roundings
  of the log-probability terms, not proportional to the KL, and its sign depends on how the
  device's `logf`/`expf` round: the MI355X baseline's scalar was 3.5e-7 to 4.2e-7 relative off
  the float64 reference in every case of the audit of the first inputs, which is what a systematic half-ulp
  `logf` bias produces in a CPU emulation (4.0e-7), while CPU float32 gives 1e-9. The default
  bound's per-element and bias tests would compare a candidate's rounding against the baseline's
  own, so a correct candidate whose exponential or logarithm rounds differently could fail; a
  whole-output SQNR does not depend on that. CPU emulations of the float32 arithmetic on the task's
  own inputs (4096 to 1M pixels per draw, n = 1 to 100 through the task generator and the
  evaluator's comparator):

  | Implementation | SQNR |
  | --- | --- |
  | the baseline's algorithm (32-lane tree reductions) | 141.9-142.6 dB |
  | `torch.log_softmax` and `xlogy` in float32, online softmax, `exp2`/`log2` formulation (with or without max subtraction), log of the softmax, no max subtraction, separate sums of `t log t` and `t log p`, `lse` added once assuming `sum(t) = 1` | 138.8-142.3 dB |
  | the baseline's algorithm with every `logf` and `expf` off by a random amount up to 2 ulp | 140.5-141.0 dB |
  | the same with a systematic half-ulp `logf` bias (the MI355X-like model) | 141.1-141.3 dB |
  | systematic 1-ulp bias in `logf`, `expf` or both, or 2-ulp bias in `logf` | 135.9-141.1 dB |
  | `shifted_16_ulp` on any of these baselines | 119.5-121.4 dB |
  | `half_written`, `fp8_rounded`, `zeros` | about 3, 32 and 0 dB |

  128 dB accepts every float32 formulation with at least 7.8 dB to spare and rejects the shift
  by at least 6.6 dB. It also rejects the shift for any baseline that passes the bound itself:
  the shift moves every element by at least 15 units of float32 roundoff (16 less the output
  rounding), so by the triangle inequality its SQNR is at most `-20 log10(15 * 2^-24 -
  10^(-128/20)) = 126.1 dB` when the baseline's own error is within 128 dB. With the first
  inputs the shift was at 120.2 dB while correct implementations reached only 127 to 134 dB and a
  1-ulp `logf` bias 121.8 dB, which is why the inputs changed. 128 dB is an error RMS of 4e-7 of
  the KL's RMS: anything computed in bfloat16 or float16 or a wrong channel fails by tens of dB,
  one zeroed pixel of average KL gives 37 dB at n=1 and 70 dB at n=2048, and NaN-poisoned
  (unwritten) pixels always fail.
- `out`: `tolerance` with `atol = 1e-6`, `rtol = 1e-4`. A bound tied to the baseline's own error
  on a single scalar (the float64 accumulator leaves only the final float32 rounding) would be
  luck. Simulated float32 accumulations of the float32 per-pixel KL at n = 16 to 2048: partial
  sums of 64 to 4096 pixels added with float32 atomics in random order stay within 7.0e-6
  relative (largest of 43 draws), a float32 pairwise sum within 1.1e-7, and the per-pixel error
  models above within 1.9e-7, so the bound accepts them with a factor of 14 to spare. One float32
  atomic per pixel into a single accumulator, like the original baseline, is off by up to 3.5e-5 at
  n=256, 1.7e-4 at n=1024 and 9.4e-4 at n=2048 and fails at the two largest batches, as the
  original baseline was judged wrong. A wrong temperature, scaling or divisor moves the result by
  far more, and a dropped image by at least 1/2048 = 4.9e-4.

The reference runs in float64 in chunks of 64 images to keep temporaries small.

Risks for the GPU audit: the SQNR figures come from a CPU emulation of the float32 arithmetic
(with modeled `logf`/`expf` biases), not from the GPU; the baseline's MI355X SQNR on `pixel_kl` is
predicted near 140 dB. The input distribution and the `pixel_kl` store are new since the last
audit; the timing is dominated by the single-address float64 atomics and should not change.
