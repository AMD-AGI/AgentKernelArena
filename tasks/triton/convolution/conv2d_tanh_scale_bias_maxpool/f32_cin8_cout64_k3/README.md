# Conv2d, tanh, scale, bias, 4x4 max pool: float32 NCHW, 8 to 64 channels, 3x3

`out = max_pool2d(tanh(conv2d(x, conv_weight, conv_bias)) * scaling_factor + bias, 4)`. The
convolution is 3x3, stride 1, no padding, no dilation, one group, with bias, from 8 to 64 channels on
a float32 NCHW batch; products and accumulation are float32 (no TF32/XF32 or other reduced
precision). `scaling_factor` is a runtime float32 scalar (default 2.0, the module's constant) and
`bias` is a learned per-channel tensor of shape `[64, 1, 1]`. The pool is a 4x4 window with stride 4
in floor mode, so the output is `[batch, 64, (height - 2) // 4, (width - 2) // 4]`, contiguous NCHW.

- **Origin.** KernelBench level 2, problem 82 (`Conv2d_Tanh_Scaling_BiasAdd_Max`), as the earlier
  AgentKernelArena task `torch2hip/kernelbench/level2/l2n82_Conv2d_Tanh_Scaling_BiasAdd_Max` used
  it: `nn.Conv2d(8, 64, 3)`, scaling 2.0, `bias = nn.Parameter(torch.randn(64, 1, 1))`,
  `nn.MaxPool2d(4)`. The input is `torch.rand` (uniform [0, 1)) as in its `get_inputs()`; the filter
  and convolution bias are drawn like `nn.Conv2d`'s default initialization, uniform within
  `±1/sqrt(8 * 9)`, and `bias` is standard normal like its `torch.randn` initialization.
- **Scenario.** Channel counts, filter size, pool size, dtype and layout are fixed; batch, height
  and width vary (separate height and width axes; the performance cases are square as in
  KernelBench, several correctness cases are not). `scaling_factor` is an input, and one
  correctness case uses -1.5: with a negative factor the window maximum comes from the smallest
  tanh, so an implementation that pools the convolution output before tanh must handle the sign.
  The factor is declared float32 and the reference applies its float32 value, as the module's
  float32 arithmetic does (2.0 and -1.5 are exact).
- **Baseline.** No production checkout has a Triton kernel for this module to vendor: eager
  PyTorch runs the convolution through MIOpen and the rest as separate elementwise and pooling
  kernels, and `torch.compile(mode="max-autotune")` can choose Inductor's `convolution2d` Triton
  template (`torch/_inductor/kernel/conv.py`) with the pointwise ops fused into its epilogue and
  the pool as a second kernel. The baseline is a standalone Triton rewrite of that design with the
  pool fused as well: one implicit-GEMM kernel whose 72-term reduction over (input channel, filter
  row, filter column), in OIHW weight order and slices of 16 (the last one half masked), gathers
  the shifted input pixels and the filter slice and accumulates `tl.dot(..., input_precision="ieee")`
  in float32 (Inductor's `ALLOW_TF32=False`). Each program covers one pooled output row (the four
  convolution rows under it) of 16 pooled columns and all 64 output channels (a 256 x 64 x 16 tile,
  8 warps, 2 stages, fixed, not autotuned); the epilogue adds the convolution bias, applies
  `libdevice.tanh` (what Inductor's pointwise codegen emits for `torch.tanh`), the scaling and the
  channel bias, reshapes the tile and takes the 4x4 window maxima in registers. Its `run()` only
  launches the kernel, so it also meets the Triton candidate rules. License: BSD-3-Clause
  (`baseline/LICENSE`, PyTorch's license file).
- **Workloads.** The five KernelBench test shapes of the earlier task (batch 32, 64, 128 at
  128x128; 64 and 128 at 256x256) plus batch 8 at 128x128 and batch 256 at 256x256, which the 100x
  span of `batch * height * width` (here 128x) requires; smaller batches would run below the 10 us
  floor. Estimated baseline time: 1.17 GFLOP for batch 8 at 128x128, at least 12 us at the
  157 TFLOP/s float32 MFMA peak plus launch (the 72-term reduction keeps the fixed tile far from
  peak), up to 152 GFLOP for batch 256 at 256x256. Correctness-only cases: the axis minimum (1x6x6,
  one 1x1 pooled output), the minimum height with a 32-column pooled row, odd sizes where floor-mode
  pooling drops the last convolution rows and columns (3x13x11), an unaligned image with 62 pooled
  columns (2x131x250), a non-square image with an odd batch whose 17 pooled columns cross a column
  tile (5x100x70), a single 128x128 image and the negative scaling factor. The largest case holds a
  537 MB input, a 260 MB output and a 520 MB float64 reference per draw; the reference convolves
  16 images at a time.
- **Comparison: `sqnr` with `min_db: 130` instead of the default `baseline_relative`.** The output
  error of a float32 convolution is dominated by accumulation rounding, which depends on the
  summation order. The element-wise default bound (each element within k times the baseline's own
  error or its row RMS) rejects correct float32 implementations that only sum in another order, and
  its per-row RMS is the single element itself when a pooled row has one column. A CPU simulation of
  this task's inputs through the evaluator's own `compare_tensor` (baseline modeled as a sequential
  float32 FMA chain in OIHW order) gave, for correct float32 variants (channel-inner tap order,
  reverse order, separate multiply and add, 2-way split-K, oneDNN, a fused multiply-add epilogue):
  violation fractions of 2-3e-2 at k = 2 on wide rows and, on one-column rows, 3e-2 even at
  `k: 8, max_violation_fraction: 0.001`. Whole-output SQNR is stable across shapes and separates the
  classes: baseline model 141.1-141.9 dB, the correct variants 139.2-146.0 dB (139.3-144.7 dB with the
  negative scaling factor), a systematic 4-ulp scaling 132 dB, the evaluator's 16-ulp shifted
  control 120.4 dB, a bf16 hi/lo split with three products 114-117 dB, TF32-like (10-bit mantissa)
  inputs 77-79 dB, fp8 rounding 31-32 dB, and a 1e-3 relative error on one corner pixel per channel
  60-93 dB or a +1e-5 error on the last pooled column 102-120 dB. Float32 Winograd convolutions,
  the fast 3x3 algorithms MIOpen uses, also pass: F(2x2, 3x3) 143.9-145.1 dB and F(4x4, 3x3)
  132.6-134.5 dB on 2x130x130. The 130 dB threshold accepts every correct float32 variant (with at
  least 9 dB of margin for direct convolution, 2.6 dB for the less accurate F(4x4, 3x3)) and
  rejects the shifted control by 10 dB and every reduced-precision scheme. Unwritten (NaN-poisoned) or non-finite elements always fail, and
  the baseline itself must reach 130 dB.
- **CPU check of the baseline.** Triton's interpreter (3.9, on CPU, with `libdevice.tanh` replaced
  by an exp-based tanh because the interpreter has no libdevice) ran the baseline against the
  reference on 6 shapes including 1x6x6, 3x13x11, 2x131x250, 5x100x70 and the negative scaling
  factor: every element written, 141.4-144.8 dB.
- **Risks for the GPU audit.** Kernels, timing and the control suite have not run on a GPU. The
  smallest performance case (batch 8 at 128x128, about 500 programs) has the thinnest margin over
  the 10 us floor. The in-register 4x4 pool reshapes the MFMA accumulator tile (a layout conversion
  through shared memory) and has not been compiled for gfx950. The SQNR margins come from the CPU
  model of the accumulation; the audit should confirm the GPU baseline's SQNR (expected about
  141 dB). The float64 reference uses PyTorch's native (im2col + GEMM) convolution because MIOpen
  has no float64 convolution.
