# Conv2d + Mish + BatchNorm2d (eval), float32 NCHW, 64 to 128 channels, 3x3

A convolution block in inference mode on a float32 NCHW image batch:

```text
y   = conv2d(x, conv_weight, conv_bias)            # 3x3, stride 1, no padding, one group
m   = y * tanh(softplus(y))                        # Mish; softplus(y) = y for y > 20
out = (m - running_mean[c]) / sqrt(running_var[c] + eps) * bn_weight[c] + bn_bias[c]
```

with 64 input channels, 128 output channels, `eps = 1e-5`, float32 convolution products and
accumulation (no TF32/XF32 or other reduced precision). The output is
`[batch, 128, height - 2, width - 2]`, contiguous NCHW.

- **Origin.** KernelBench level 2, problem 52 (`Conv2d_Activation_BatchNorm`), in the variant the
  earlier AgentKernelArena KernelBench task used (`nn.Conv2d(64, 128, 3)` followed by the
  activation and `nn.BatchNorm2d(128)`), evaluated in eval mode as that task did. The input is
  `torch.rand` (uniform [0, 1)) as in its `get_inputs()`; the convolution weight and bias are drawn
  like `nn.Conv2d`'s default init (uniform within `±1/sqrt(64 * 9) = ±1/24`). A freshly built
  `BatchNorm2d` has identity statistics (mean 0, variance 1, scale 1, shift 0), which would leave
  most of the batch-norm math unchecked, so the task draws trained-looking statistics instead:
  `running_mean ~ N(0, 0.2²)`, `running_var ~ U(0.05, 0.5)`, `bn_weight ~ U(0.5, 1.5)`,
  `bn_bias ~ N(0, 0.2²)`. With these, dropping `eps` alone lowers the SQNR to 86-89 dB.
- **Scenario.** Channel counts, filter size, stride, padding, `eps`, dtype and layout are fixed;
  batch, height and width vary. Height and width are separate axes; the performance cases are
  square as in KernelBench, and two correctness cases are not.
- **Baseline.** There is no Triton implementation of this block in a production checkout to vendor:
  eager PyTorch runs MIOpen plus pointwise kernels, and `torch.compile(mode="max-autotune")`
  benchmarks MIOpen against Inductor's `convolution2d` Triton template
  (`torch/_inductor/kernel/conv.py`) with the pointwise tail fused into the template epilogue. The
  baseline is a standalone Triton rewrite of that fused template (non-unrolled loop for non-1x1
  kernels, groups 1): an implicit GEMM over `[batch * out_h * out_w, 128]` that, for each of the 9
  filter taps and each 32-channel slice, accumulates `tl.dot(..., input_precision="ieee")` in
  float32, then applies Inductor's lowering of the tail in float32 — the peeled-off conv bias,
  `where(v > 20, v, log1p(exp(v)))`, `tanh(.) * v`, and the eval-mode batch-norm decomposition
  `(v - mean) * rsqrt(var + eps) * weight + bias` (`torch/_decomp/decompositions.py`) — before the
  strided NCHW store. Jinja placeholders became kernel arguments and constexprs, and the autotuned
  tile configuration is fixed to one of Inductor's ROCm conv configs (128x128x32, 2 stages, 4 warps,
  `matrix_instr_nonkdim=16`); no autotuning ran, so it is likely slower than the best autotuned
  choice. Inductor's `INDEX_DTYPE` choice is kept as an int64 switch for tensors of 2**31 or more
  elements (the declared domain stays below it). License: BSD-3-Clause (`baseline/LICENSE`).
- **Workloads.** The performance cases are the five KernelBench test shapes of the earlier task,
  batch 16 at 64x64 to batch 64 at 128x128, plus batches 2 and 4 at 64x64 so the batch x height x
  width sizes span 128x. Estimated baseline time: 1.13 GFLOP for batch 2 at 64x64 (about 20-40 us at
  30-60 TFLOP/s of float32 MFMA) up to 150 GFLOP for batch 64 at 128x128. A single 64x64 image and
  the boundary cases are correctness-only: batch 1 (axis minimum), an odd batch whose pixel count
  leaves a partial tile, 100x75 and 34x130 images (non-power-of-two, not a multiple of 16) and a 3x5
  image (the minimum height, a 1x3 output). The largest case holds a 268 MB input, a 520 MB output
  and a 1 GB float64 reference per draw.
- **Comparison: `sqnr` with `min_db: 124` instead of the default `baseline_relative`.** The error of
  a float32 convolution with 576-term dot products is dominated by accumulation rounding, which
  depends on summation order. The element-wise default bound (each element within k times the
  baseline's own error or its row RMS) therefore rejects correct float32 implementations that sum in
  a different order: a CPU simulation of this task's inputs through the evaluator's own
  `compare_tensor` (baseline modeled as tap-major accumulation in groups of four products, like fp32
  MFMA) gave violation fractions for correct sequential FMA loops of up to 1.7e-2 at k = 6 and up to
  8.4e-3 at k = 8 on small images, so even `k: 8, max_violation_fraction: 0.01` leaves no margin
  while admitting 1 % of arbitrarily wrong elements. Whole-output SQNR separates the classes:
  baseline model 134-136 dB, oneDNN 134-136 dB, 4-way split-K 135-136 dB, Winograd F(2x2,3x3)
  136-137 dB, fast-math epilogues (`log(1 + exp)`, `tanh` via `exp`, `1 / sqrt`) and batch norm
  folded into a per-channel scale and shift 134-136 dB. Every sequential direct order (one float32
  accumulator, FMA or multiply-then-add, tap-major, channel-major or reversed, bias first or last)
  averages 129.9 dB, with a spread that grows as the output shrinks: over 1,600 draws of the
  768-element `b2_h3_w5` output it ranged from 125.6 to 133.4 dB (standard deviation 1.0 dB, 1st
  percentile 127.4 dB), and over 200 draws of a 32,768-element output from 127.6 to 131.5 dB.
  Rejected classes: Winograd F(4x4,3x3) 124.0-125.1 dB, the evaluator's 16-ulp shifted control
  119.7-120.4 dB (a 16-ulp scale error alone caps the SQNR at 120.4 dB), bf16x2 products 110-111.5
  dB, a missing `eps` 85-89 dB, TF32-like (10-bit mantissa) inputs 74-75 dB, a 1e-3 relative error
  on one corner pixel per channel 90-96 dB and on the last column 75-78 dB. The 124 dB threshold
  (10-12 dB below the baseline model) accepts every simulated direct float32 implementation: it lies
  about 6 standard deviations below their mean on the smallest output, whose worst of 1,600 draws
  stays 1.5 dB above it, and larger outputs stay at least 3.6 dB above it. It rejects the shifted
  control by at least 3.6 dB and every reduced-precision scheme. A 126 dB threshold would reject a
  correct sequential loop on about one draw in a thousand of the smallest case. Winograd F(4x4,3x3)
  sits at the threshold and is not guaranteed to pass. Unwritten (NaN-poisoned) or non-finite
  elements always fail. The baseline itself must also reach 124 dB. Against the earlier task's
  `allclose(rtol=1e-4, atol=1e-5)`, which let every element carry several hundred times the float32
  accumulation noise, the bound is far stricter on error spread over the output and on every small
  case (one wrong element of `b2_h3_w5` may be off by at most about 1e-5), while in the largest
  output a single isolated element could be off by up to about 4e-3 and still pass.
- **Risks for the GPU audit.** Kernels, timing and the control suite have not run on a GPU; the
  baseline's indexing, masking, partial tiles and epilogue were checked only in Triton's CPU
  interpreter (SQNR 137-138 dB on small odd shapes, with `libdevice` stood in by core ops there).
  Batch 2 at 64x64 runs only 61 tiles (fewer than the 256 CUs), each with 18 K steps of 128x128x32,
  so even at its CU's full float32 MFMA rate (157 TFLOP/s per GPU) it takes about 30 us; the
  epilogue uses `triton.language.extra.libdevice` (`log1p`, `tanh`, `rsqrt`). The float64 reference
  uses PyTorch's native (im2col + GEMM) convolution because MIOpen has no float64 convolution.
