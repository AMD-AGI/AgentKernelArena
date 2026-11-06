# Fused bias + leaky ReLU, float32 NCHW, 256 channels

StyleGAN2's `FusedLeakyReLU` (`out = scale * leaky_relu(x + bias[c], negative_slope)`) on a
`[n, 256, 128, 128]` float32 feature map with the StyleGAN2 defaults `negative_slope = 0.2`,
`scale = sqrt(2)`. The per-channel bias is drawn from N(0, 0.5^2) so a missing or mis-indexed bias is
visible (the old task's zero bias hid that). The reference is float64 and chunked over images to bound
temporaries.

- **Baseline.** The HIP implementation provided by the old `hip2hip/FusedLeakyReLU` task (identical to
  its initial candidate), a hand-written task-original kernel, not an upstream production kernel. It is
  reduced to float32, the pybind module is replaced by the `run` torch op, and the output is written into
  the preallocated tensor instead of being allocated in the launcher. Dispatch is unchanged: a 2D grid
  over (spatial blocks, n * channels) while n * channels <= 65535, otherwise a 1D grid-stride kernel.
  The largest case (n = 256, 65536 slices) therefore runs the 1D kernel; both paths are timed.
- **Workloads.** Batch sizes 1 to 256 at 256 channels and 128x128 (StyleGAN2's 256-channel level).
  Performance cases are n = 2 to 256 (128x span); each image moves 33.5 MB, estimated at least
  15 us at n = 2. Single images and small odd batches are correctness-only boundary cases. The largest
  case needs 4.3 GB each for input and output and an 8.6 GB float64 reference.
- **Slope/scale cases.** Four correctness-only cases reuse the (negative_slope, scale) pairs of the original
  task's test cases (0.1/0.5, 0.35/1.25, 0.5/2.0, 0.75/3.0) at 256 channels; its smaller channel counts and
  spatial sizes belong to other scenarios.
- **Comparison: `sqnr` with `min_db: 135` instead of the default `baseline_relative`.** The operator is
  elementwise and exact up to float32 rounding, but the element-wise bound is calibrated to the
  baseline's own rounding pattern. The baseline computes `max(v,0)*scale + min(v,0)*(slope*scale)` with
  the product `slope*scale` pre-rounded, so many of its elements are within a fraction of an ulp of the
  float64 reference and the per-element bound is tight there. A correct plain float32 evaluation
  (`(x+bias)`, `*slope`, `*scale`: three roundings on the negative side, a different association)
  differs from the baseline's rounding on a small fraction of elements (violation fractions 3e-6 to 3.4e-4
  for the default and the correctness-only slope/scale pairs, peak 1.4x the bound) and, because the
  float32 constants and chained roundings are a systematic effect, carries a measured bias z-score of
  0.03 to 0.08 of its RMS. The comparator's bias test (threshold `5/sqrt(n)`, 0.0024 for 4M elements)
  rejects that, so the n1/slope 0.35/scale 1.25 case failed in the QA audit's float32 evaluation although
  the violation fraction was within `max_violation_fraction`. Whole-output SQNR is insensitive to this
  rounding-pattern difference. Measured CPU emulation (n = 1 and 3, all five slope/scale pairs):
  float32 evaluation 147.4 to 151.5 dB, baseline arithmetic 147.7 to 151.4 dB, the auditor's
  `shifted_16_ulp` control (output scaled by `1 + 16u`) 120.4 dB, zeros 0 dB. 135 dB leaves 12 dB
  below the weakest correct evaluation and 15 dB above the shift. Any wrong or missing bias, wrong slope or
  scale, unwritten, stale or partly written output, or a wrong element in either grid path lands far below
  135 dB (a single element off by one among 4M elements is about 66 dB).
- **Risk.** HIP build and timing are unverified on a GPU: the CPU pre-check cannot compile `kernel.hip`.
