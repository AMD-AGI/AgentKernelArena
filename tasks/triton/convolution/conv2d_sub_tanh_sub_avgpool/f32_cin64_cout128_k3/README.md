# Conv2d, subtract, tanh, subtract, 2x2 average pool: float32 NCHW, 64 to 128 channels, 3x3

`out = avg_pool2d(tanh(conv2d(x, conv_weight, conv_bias) - subtract1_value) - subtract2_value, 2)`.
The convolution is 3x3, stride 1, no padding, no dilation, one group, with bias, from 64 to 128
channels on a float32 NCHW batch; products and accumulation are float32 (no TF32/XF32 or other
reduced precision). The pool is a 2x2 window with stride 2 in floor mode, so the output is
`[batch, 128, (height - 2) // 2, (width - 2) // 2]`, contiguous NCHW. `subtract1_value` and
`subtract2_value` are runtime float32 scalars (defaults 0.5 and 0.2, the module's constants).

- **Origin.** KernelBench level 2, problem 46 (`Conv2d_Subtract_Tanh_Subtract_AvgPool`), as the
  earlier AgentKernelArena task `torch2hip/kernelbench/level2/l2n46_Conv2d_Subtract_Tanh_Subtract_AvgPool`
  used it: `nn.Conv2d(64, 128, 3)`, subtractions 0.5 and 0.2, `nn.AvgPool2d(2)`. The input is
  `torch.rand` (uniform [0, 1)) as in its `get_inputs()`; the filter and bias are drawn like
  `nn.Conv2d`'s default initialization, uniform within `±1/sqrt(64 * 9) = ±1/24`. With these
  inputs the convolution output is of order 0.5, so tanh runs on both sides of its linear range.
- **Scenario.** Channel counts, filter size, pool size, dtype and layout are fixed; batch, height
  and width vary (separate height and width axes; the performance cases are square as in
  KernelBench, several correctness cases are not). The two subtraction constants are inputs, and
  one correctness case uses other values (-0.3 and 0.7) so they cannot be hard-coded. They are
  declared float32 and the reference applies their float32 values, as the module's float32
  arithmetic does (`x - 0.2` on a float32 tensor subtracts float32(0.2)); applying the double
  values -0.3 and 0.7 instead would put a 2.2e-8 offset on every element of an otherwise exact
  float32 result, a 146 dB SQNR floor of its own.
- **Baseline.** No production checkout has a Triton kernel for this module to vendor: eager
  PyTorch runs the convolution through MIOpen and the rest as separate elementwise and pooling
  kernels, and `torch.compile(mode="max-autotune")` can choose Inductor's `convolution2d` Triton
  template (`torch/_inductor/kernel/conv.py`) with the pointwise ops fused into its epilogue and
  the pool as a second kernel. The baseline is a standalone Triton rewrite of that design with the
  pool fused as well: one implicit-GEMM kernel whose reduction over (input channel, filter row,
  filter column), in OIHW weight order, gathers the shifted input pixels and the filter slice and
  accumulates `tl.dot(..., input_precision="ieee")` in float32 (Inductor's `ALLOW_TF32=False`).
  Each program covers one pooled output row (the two convolution rows under it) of 32 pooled
  columns and 64 output channels (a 128 x 64 x 32 tile, 4 warps, 2 stages, fixed, not autotuned);
  the epilogue adds the bias, applies both subtractions and `libdevice.tanh` (what Inductor's
  pointwise codegen emits for `torch.tanh`), reshapes the tile and averages the 2x2 windows in
  registers. Its `run()` only launches the kernel, so it also meets the Triton candidate rules.
  License: BSD-3-Clause (`baseline/LICENSE`, PyTorch's license file).
- **Workloads.** The five KernelBench test shapes of the earlier task (batch 32, 64, 128 at 64x64;
  64 and 128 at 128x128) plus batch 4 at 64x64 and batch 2 at 128x128, which the 100x span of
  `batch * height * width` (here 128x) requires. Estimated baseline time: 2.3 GFLOP for batch 4 at
  64x64, at least 19 us at the 157 TFLOP/s float32 MFMA peak plus launch, up to 300 GFLOP (about
  2 ms at peak) for batch 128 at 128x128. Correctness-only cases: the axis minimum (1x4x4, one 1x1
  pooled output), the minimum height with a 32-column pooled row, odd sizes where floor-mode pooling
  drops the last convolution row and column (3x7x9), an unaligned image whose 65 pooled columns
  cross two column tiles (2x67x133), a non-square image with an odd batch (5x100x37), the largest
  declared image (256x256) and the scalar override. The largest case holds a 537 MB input, a 260 MB
  output and a 520 MB float64 reference per draw; the reference convolves 8 images at a time.
- **Comparison: `sqnr` with `min_db: 130` instead of the default `baseline_relative`.** The output
  error of a float32 convolution with 576-term dot products is dominated by accumulation rounding,
  which depends on the summation order. The element-wise default bound (each element within k times
  the baseline's own error or its row RMS) rejects correct float32 implementations that only sum in
  another order, and its per-row RMS is the single element itself when a pooled row has one column.
  A CPU simulation of this task's inputs through the evaluator's own `compare_tensor` (baseline
  modeled as a sequential float32 FMA chain in OIHW order) gave, for correct float32 variants
  (channel-inner tap order, reverse order, separate multiply and add, 2-way split-K, oneDNN, the bias
  folded into `subtract1_value`): violation fractions of 1-2e-2 at k = 2 on wide rows and, on
  one-column rows, 2e-2 even at `k: 8, max_violation_fraction: 0.001`. Whole-output SQNR is
  stable across shapes and separates the classes: baseline model 140.8-141.9 dB, the correct
  variants 139.5-145.0 dB (137.5-141.4 dB with the alternative scalars, whose output has less signal),
  a systematic 4-ulp scaling 132 dB, the evaluator's 16-ulp shifted control 120.4 dB, a bf16 hi/lo
  split with three products 115-120 dB, TF32-like (10-bit mantissa) inputs 78-82 dB, fp8 rounding
  32 dB, and a 1e-3 relative error on one corner pixel per channel 60-93 dB or a +1e-5 error on the
  last pooled column 96-115 dB. The alternative-scalar figures were measured against a reference
  that applied the double scalar values; with the float32 scalars (above) a recheck on 1x4x4, 3x7x9
  and 2x34x34 gave 142.8-143.7 dB for the baseline and 137.1-142.6 dB for oneDNN's float32
  convolution. Float32 Winograd convolutions, the fast 3x3 algorithms MIOpen uses, also pass:
  F(2x2, 3x3) 142.9-143.2 dB and F(4x4, 3x3) 134.8-135.0 dB on 2x66x66. The 130 dB threshold
  accepts every correct float32 variant with at least 4.8 dB of margin (7 dB for direct
  convolution) and rejects the shifted control by 10 dB and every reduced-precision scheme.
  Unwritten (NaN-poisoned) or non-finite elements always fail, and the baseline itself must reach
  130 dB.
- **CPU check of the baseline.** Triton's interpreter (3.9, on CPU, with `libdevice.tanh` replaced
  by an exp-based tanh because the interpreter has no libdevice) ran the baseline against the
  reference on 8 shapes including 1x4x4, 3x7x9, 2x67x133 and 5x100x37: every element written,
  141.6-145.5 dB; after the reference change to float32 scalars, 142.8-145.5 dB on 1x4x4, 3x7x9 and
  2x34x34 with both scalar pairs.
- **Risks for the GPU audit.** Kernels, timing and the control suite have not run on a GPU. The
  in-register 2x2 pool reshapes the MFMA accumulator tile (a layout conversion through shared
  memory) and has not been compiled for gfx950. The floor estimate for batch 4 at 64x64 assumes
  the fixed tile does not run near the float32 MFMA peak. The SQNR margins come from the CPU model
  of the accumulation; the audit should confirm the GPU baseline's SQNR (expected about 141 dB).
  The float64 reference uses PyTorch's native (im2col + GEMM) convolution because MIOpen has no
  float64 convolution.
