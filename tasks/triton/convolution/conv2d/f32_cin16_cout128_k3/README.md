# Conv2d, float32 NCHW, 16 to 128 channels, 3x3

A standard two-dimensional convolution, `out = conv2d(x, weight)`, on a float32 NCHW image batch
with an OIHW 3x3 filter bank: 16 input channels, 128 output channels, stride 1, no padding, no
dilation, one group, no bias. Products and accumulation are float32 (no TF32/XF32 or other reduced
precision). The output is `[batch, 128, height - 2, width - 2]`, contiguous NCHW.

- **Origin.** KernelBench level 1, problem 63 ("conv standard 2D, square input, square kernel"),
  `nn.Conv2d(16, 128, 3, bias=False)` in the variant the earlier AgentKernelArena KernelBench task
  used. The input is `torch.rand` (uniform [0, 1)) as in its `get_inputs()`; the filters are drawn
  like `nn.Conv2d`'s default init, uniform within `±1/sqrt(16 * 9) = ±1/12`.
- **Scenario.** Channel counts, filter size, stride, padding, dtype and layout are fixed; the batch,
  height and width vary. Height and width are separate axes (production convolutions are not
  always square, and their product is the work that varies); the performance cases are square as in
  KernelBench, and two correctness cases are not.
- **Baseline.** There is no Triton convolution in a production checkout to vendor: PyTorch runs
  this layer through MIOpen, and `torch.compile(mode="max-autotune")` benchmarks MIOpen against
  Inductor's `convolution2d` Triton template (`torch/_inductor/kernel/conv.py`). The baseline is a
  standalone Triton rewrite of that template (its non-unrolled loop for non-1x1 kernels, groups 1):
  an implicit GEMM over `[batch * out_h * out_w, 128]` that, for each of the 9 filter taps and each
  16-channel slice, gathers the shifted input pixels and the filter slice and accumulates
  `tl.dot(..., input_precision="ieee")` in float32, then stores the NCHW tile. Jinja placeholders
  became kernel arguments and constexprs, `store_output` became a strided store, and the autotuned
  tile configuration is fixed to one of Inductor's ROCm conv configs (128x128x16, 2 stages, 4 warps,
  `matrix_instr_nonkdim=16`); no autotuning ran, so it is likely slower than the best autotuned
  choice and than MIOpen. Inductor's `INDEX_DTYPE` choice is kept as an int64 switch for tensors of
  2**31 or more elements (the declared domain stays below it). License: BSD-3-Clause
  (`baseline/LICENSE`).
- **Workloads.** The performance cases are the five KernelBench test shapes of the earlier task,
  batch 4 at 128x128 to batch 16 at 512x512, plus batch 2 at 128x128 so the batch x height x width
  sizes span 128x. Estimated baseline time: 1.17 GFLOP for batch 2 at 128x128 (about 20-40 us at
  30-60 TFLOP/s of float32 MFMA; at the full 157 TFLOP/s float32 MFMA rate its single wave of 248
  tiles would still take 7.7 us of compute plus the launch floor) up to 153 GFLOP for batch 16 at
  512x512. A single 128x128 image and the boundary cases are correctness-only: batch 1 (axis
  minimum), an odd batch whose pixel count leaves a partial tile, 100x75 and 34x130 images
  (non-power-of-two, not a multiple of 16) and a 3x5 image (the minimum height, a 1x3 output). The
  upstream KernelBench shape (16, 16, 1024, 1024) is outside the declared domain: its 2.1e9-element
  output needs 64-bit indexing and a 17 GB float64 reference per draw. The largest case holds a 268
  MB input, a 2.1 GB output and a 4.3 GB float64 reference per draw.
- **Comparison: `sqnr` with `min_db: 129` instead of the default `baseline_relative`.** The output
  error of a float32 convolution with 144-term dot products is dominated by accumulation rounding,
  which depends on summation order, not by the final rounding. The element-wise default bound (each
  element within k times the baseline's own error or its row RMS) therefore rejects correct float32
  convolutions that merely sum in a different order. A CPU simulation of this task's inputs through
  the evaluator's own `compare_tensor` (baseline modeled as tap-major accumulation in groups of four
  products, like fp32 MFMA) gave, for correct float32 implementations (sequential FMA loops in
  channel-major, tap-major and reverse order, oneDNN, 4-way split-K): violation fractions of 1.5e-1
  at k = 2, 2-4e-2 at k = 4 and up to 7e-3 at k = 8 on small images, so even `k: 8,
  max_violation_fraction: 0.01` leaves no margin and would admit 1 % of arbitrarily wrong elements.
  Whole-output SQNR is stable across shapes and seeds and separates the classes cleanly: baseline
  model 139 dB, Winograd F(2x2,3x3) 138 dB, split-K 139 dB, every sequential direct order (one
  float32 accumulator, FMA or multiply-then-add, tap-major, channel-major or reversed) 133.3 dB on
  average, with standard deviation 0.44 dB and a lowest value of 131.8 dB over 1,300 draws of the
  768-element `b2_h3_w5` output, and 132.6-133.9 dB over 200 draws of a 32,768-element output;
  Winograd F(4x4,3x3) 125.6 dB (5x the baseline's noise), the evaluator's 16-ulp shifted control
  120.2 dB, bf16x2 products 108 dB, TF32-like (10-bit mantissa) inputs 71 dB, a 1e-3 relative error
  on one corner pixel per channel 96-108 dB and on the last column 78-84 dB. The 129 dB threshold
  (10 dB below the baseline model) accepts any direct float32 convolution, about 4 dB below its mean
  and 2.8 dB below its worst simulated draw of the smallest output, and rejects the shifted control
  with 9 dB and Winograd F(4x4,3x3) (2.6x the noise of a sequential direct loop) with 3.4 dB, as
  well as every reduced-precision product scheme. Unwritten (NaN-poisoned) or non-finite elements
  always fail. The baseline itself must also reach 129 dB. Against the earlier task's
  `allclose(rtol=1e-4, atol=1e-5)`, which let every element carry several hundred times the float32
  accumulation noise, the bound is far stricter on error spread over the output and on every small
  case (one wrong element of `b2_h3_w5` may be off by at most about 3e-6), while in the largest
  output a single isolated element could be off by up to about 2e-3 and still pass.
- **Risks for the GPU audit.** Kernels, timing and the control suite have not run on a GPU; the
  baseline's indexing, masking and partial tiles were checked only in Triton's CPU interpreter (SQNR
  140-141 dB on small odd shapes). Batch 2 at 128x128 is the shortest performance case: one wave of
  248 tiles, each with 9 K steps of 128x128x16, about 7.7 us of compute at the CU's full float32
  MFMA rate plus the launch floor, so it should stay above 10 us. The float64 reference uses
  PyTorch's native (im2col + GEMM) path because MIOpen has no float64 convolution.
