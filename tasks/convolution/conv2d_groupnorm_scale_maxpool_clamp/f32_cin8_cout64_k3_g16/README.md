# conv2d + GroupNorm + scale + MaxPool + clamp, float32, 8 -> 64 channels, 3x3, 16 groups

## Operator

For float32 NCHW images `x[batch, 8, height, width]`:

1. `y = conv2d(x, conv_weight[64, 8, 3, 3], conv_bias[64])`: valid (unpadded) stride-1
   convolution, so `y` is `[batch, 64, height - 2, width - 2]`.
2. GroupNorm with 16 groups of 4 channels: per sample and group, the mean and the biased
   variance are taken over the 4 channels and the **whole** convolution output (including the
   rows and columns that pooling later drops);
   `y = (y - mean) * rsqrt(var + epsilon) * gn_weight + gn_bias`.
3. `y = y * scale` with `scale` of shape `[64, 1, 1]`. The scale is applied **before** pooling;
   about 30% of the channels have a negative scale, for which `max(s * v) != s * max(v)`.
4. `max_pool2d` with a 4x4 window and stride 4, no padding, floor mode: the output is
   `[batch, 64, (height - 2) // 4, (width - 2) // 4]`.
5. `clamp(y, clamp_min, clamp_max)` with the defaults 0 and 1.

`epsilon` (1e-5), `clamp_min` and `clamp_max` are scalar inputs.

## Origin and scenario

This is KernelBench level-2 problem 85 (`Conv2d_GroupNorm_Scale_MaxPool_Clamp`), the former
`torch2hip/kernelbench/level2/l2n85_Conv2d_GroupNorm_Scale_MaxPool_Clamp` task. The module
fixes `in_channels=8, out_channels=64, kernel_size=3, num_groups=16, scale_shape=(64, 1, 1),
maxpool_kernel_size=4, clamp=[0, 1]`; batch and image size vary. The scenario keeps those
constants and float32 contiguous NCHW layout; `batch`, `height` and `width` are var axes.

## Inputs

- `x` is uniform on [0, 1), as KernelBench's `torch.rand`.
- `conv_weight` and `conv_bias` are uniform on `[-1/sqrt(72), 1/sqrt(72)]`, the range of
  PyTorch's default `nn.Conv2d` initialization for this fan-in.
- KernelBench initializes `gn_weight = 1`, `gn_bias = 0` and `scale = 1`. Constant draws are
  degenerate for the QA auditor and saturate the clamp for about half of the outputs, so the
  task draws `gn_weight ~ U[0.2, 0.5]`, `gn_bias ~ N(0.3, 0.15)` and `scale ~ U[-0.4, 1.0]`.
  About 75-80% of the outputs then fall strictly inside (0, 1), 15-25% clamp at 0 and 2-5% at 1,
  and negative-scale channels check the order of scaling and pooling.

## Workloads

Performance cases are the KernelBench benchmark inputs `(16, 64x64)`, `(32, 128x128)`,
`(64, 128x128)`, `(128, 128x128)` (the problem's own shape) and `(64, 256x256)`, plus
`(128, 256x256)` so that the batch x height x width product spans 128x. The smallest is
estimated at roughly 15-30 us on MI355X (about 50 MB of float32 traffic over three kernels
plus the masked fp32 MFMA convolution); the largest at a few milliseconds.

Correctness-only cases cover the smallest input (one image of 6x6, one pooled output per
channel), a single image at the smallest height, an odd batch, non-square images whose sizes
are not multiples of 16 or of the pooling window, and a cropped pooling tail.

## Comparison bound

`out` uses `sqnr` with `min_db: 126` instead of the default `baseline_relative` bound. The
output is float32 and its error is not output rounding but the float32 accumulation error of
the convolution amplified by the normalization: several ulp, roughly constant in absolute terms
rather than proportional to the output. Under `baseline_relative` (per-element multiples of the
baseline's own error, floored by the per-row RMS of the short 15-63-element pooled rows, many of
them exactly clamped) a correct implementation with a different accumulation order violates the
bound on about 1% of the elements, and the bound's bias test, whose effective sample count is
the number of GroupNorm groups rather than of elements, rejects correct implementations on
small batches a few percent of the time.

CPU emulations of float32 implementations against the float64 reference give 136-143 dB for
correct implementations with any convolution accumulation order (sequential, blocked by 4 as
fp32 MFMA does, blocked by 32, PyTorch's CPU convolution) and sum/sum-of-squares or Welford
statistics, on every workload shape including the 6x6 image; an emulation of the baseline's
own arithmetic gives 141 dB. Combining the 1024 per-lane Welford partials sequentially instead
of as a tree (as the Triton CPU interpreter does) gives 132-142 dB and still passes. The
`shifted_16_ulp` control (every output multiplied by `1 + 16 * 2^-24`) gives at most 120.4 dB
whatever the baseline's accuracy, so 126 dB rejects it with 5.6 dB to spare. The NaN, zero,
stale, half-written and fp8-rounded controls fail by tens of dB, and in the largest case a
single output wrong by more than about 1e-3 already fails (far less in the small cases).
TF32 or bf16 inputs, or a 16-bit intermediate, fall far below 126 dB.

## Baseline

There is no production Triton kernel for this module. The baseline is a standalone Triton
rewrite of what TorchInductor (`torch.compile`, max-autotune with the Triton convolution
template) generates for it, from PyTorch commit `d0f2af8b66b20da4baa50644c0cd3f887a2eae2e`:

- `_conv2d_bias_kernel`: Inductor's `conv2d_template` (`torch/_inductor/kernel/conv.py`)
  rendered for contiguous NCHW, stride 1, no padding and one group, with the bias fused into
  the store and `input_precision="ieee"` fp32 dots. The block configuration
  (`BLOCK_M=128, BLOCK_N=64, BLOCK_K=16`, 4 warps, 2 stages) is fixed instead of autotuned:
  it is the ROCm `ConvConfig(128, 128, 32, 2, 8)` as Inductor scales it to 64 output and 8
  input channels and caps it at 4 warps (the non-unrolled loop of a non-1x1 kernel
  miscompiles with 8, triton#1254). With 8 input channels, half of each 16-wide K block is
  masked.
- `_group_norm_stats_kernel`: one program per (sample, group) runs Inductor's looped
  `welford_reduce` and `welford_combine` (`torch/_inductor/runtime/triton_helpers.py`) and
  stores the mean and `rsqrt(var + eps)`.
- `_norm_scale_pool_clamp_kernel`: one fused pointwise kernel per pooled output that normalizes
  the 16 window elements, applies the affine and the scale, takes the NaN-propagating maximum
  and clamps, both with Inductor's `maximum`/`minimum` helpers
  (`tl.maximum`/`tl.minimum` with `propagate_nan=ALL`).

The convolution result is staged in a float32 temporary allocated in `run()`. PyTorch is
BSD-3-Clause; `baseline/LICENSE` is its license.

## GPU risks (not yet run)

- Timing: the smallest performance case is estimated well above 10 us; not measured.
- Numerics: the 126 dB threshold was calibrated with CPU emulations of float32 accumulation
  orders, not with the real fp32 MFMA rounding; the audit's baseline and control SQNRs should
  confirm it.
- The baseline was checked for correctness only with the Triton CPU interpreter on small shapes
  (SQNR 132-142 dB against the reference there, with NumPy's float32 matmul in place of MFMA
  and a sequential combine of the Welford partials).
