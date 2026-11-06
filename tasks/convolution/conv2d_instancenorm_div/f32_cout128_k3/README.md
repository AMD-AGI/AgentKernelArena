# conv2d + InstanceNorm2d + divide, float32, 64 -> 128 channels, 3x3

## Operator

For float32 NCHW images `x[batch, 64, height, width]`:

1. `y = conv2d(x, conv_weight[128, 64, 3, 3], conv_bias[128])`: valid (unpadded) stride-1
   convolution, so `y` is `[batch, 128, height - 2, width - 2]`.
2. `InstanceNorm2d(128)` with its defaults (`affine=False`, `track_running_stats=False`):
   every (sample, channel) plane is normalized by its own mean and biased variance,
   `y = (y - mean) * rsqrt(var + epsilon)`.
3. `out = y / divide_by`.

`epsilon` (1e-5) and `divide_by` (2.0) are scalar inputs. The convolution bias cancels
mathematically in the normalization; an implementation may exploit that, but it is part of
the interface. A 3x3 image gives a single-element plane, which `nn.InstanceNorm2d` rejects;
the reference (and the baseline) then output 0. No workload uses that shape.

## Origin and scenario

This is KernelBench level-2 problem 17 (`Conv2d_InstanceNorm_Divide`), the former
`torch2hip/kernelbench/level2/l2n17_Conv2d_InstanceNorm_Divide` task. The module fixes
`in_channels=64, out_channels=128, kernel_size=3, divide_by=2.0`; batch and image size vary.
The scenario keeps those constants and float32 contiguous NCHW layout; `batch`, `height` and
`width` are var axes.

## Inputs

- `x` is standard normal. KernelBench draws `torch.rand` (uniform on [0, 1)). With that
  positive mean every convolution output carries a large common offset that the normalization
  removes again, so the float32 error relative to the normalized output grows about 2x: a
  correct implementation that accumulates the 576-term convolution sequentially then reaches
  only about 120-121 dB SQNR, no better than the `shifted_16_ulp` control (120.4 dB), and no
  bound could tell them apart. Zero-mean activations (as after a normalization layer) keep the
  operator unchanged and separate the two by about 7 dB.
- `conv_weight` and `conv_bias` are uniform on `[-1/24, 1/24]`, the range of PyTorch's default
  `nn.Conv2d` initialization for fan-in 576.

## Workloads

Performance cases are the KernelBench benchmark inputs `(32, 64x64)`, `(64, 64x64)`,
`(128, 64x64)`, `(64, 128x128)` and `(128, 128x128)` (the problem's own shape), extended down to
inference batches `(4, 64x64)` and `(16, 64x64)` so that the batch x height x width product
spans 128x. The smallest is 2.3 GFLOP of fp32 convolution, estimated at a few tens of us on
MI355X; the largest is 300 GFLOP, a few milliseconds.

Correctness-only cases cover a single image at the smallest height (`height = 3`, one output
row per plane), an odd batch, a non-square image whose sizes are not multiples of 16, and an
odd-width image.

## Comparison bound

`out` uses `sqnr` with `min_db: 124` instead of the default `baseline_relative` bound. The
output is float32 and its error is not output rounding but the float32 accumulation error of
the 576-term convolution amplified by the normalization, several ulp of the output's RMS.
Under `baseline_relative`, a correct implementation with a different accumulation order
violates the default per-element bound on up to a few percent of the elements, and because
the output has zero mean in every plane, a relative shift creates no mean error: the
`shifted_16_ulp` control could only be caught element by element, with a margin that depends
on how accurately the baseline happens to accumulate.

CPU emulations of float32 implementations against the float64 reference give, on every
workload shape down to 2x2 planes:

| Implementation | SQNR |
| --- | --- |
| single fp32 accumulator, sequential over the 576 terms | 126-127.4 dB |
| PyTorch's CPU fp32 convolution | 126-133 dB |
| accumulation in blocks of 4 (fp32 MFMA, the baseline's arithmetic) | 132-133 dB |
| accumulation in blocks of 32 | 134.6-136.6 dB |
| `shifted_16_ulp` control | 119.5-120.4 dB |

124 dB therefore accepts every correct float32 implementation with at least 2 dB to spare and
rejects the 16-ulp shift with at least 3.6 dB; the shifted control can never exceed 120.4 dB,
whatever the baseline. The NaN, zero, stale, half-written and fp8-rounded controls fail by tens
of dB, and in the largest case a single output wrong by more than about 5e-3 already fails
(far less in the small cases). TF32 or bf16 inputs, or a 16-bit intermediate, fall far below
124 dB.

## Baseline

There is no production Triton kernel for this module. The baseline is a standalone Triton
rewrite of what TorchInductor (`torch.compile`, max-autotune with the Triton convolution
template) generates for it, from PyTorch commit `d0f2af8b66b20da4baa50644c0cd3f887a2eae2e`:

- `_conv2d_bias_kernel`: Inductor's `conv2d_template` (`torch/_inductor/kernel/conv.py`)
  rendered for contiguous NCHW, stride 1, no padding and one group, with the bias fused into
  the store and `input_precision="ieee"` fp32 dots. The block configuration
  (`BLOCK_M=128, BLOCK_N=128, BLOCK_K=32`, 2 stages) is fixed instead of autotuned. It is the
  ROCm `ConvConfig(128, 128, 32, 2, 8)` with 4 warps, because Inductor's conv lowering caps the
  non-unrolled loop of a non-1x1 kernel at 4 warps (it miscompiles with 8, triton#1254).
  The result is written to `out`.
- `_instance_norm_div_kernel`: one program per (sample, channel) plane runs Inductor's looped
  `welford_reduce` and `welford_combine` (`torch/_inductor/runtime/triton_helpers.py`), then
  normalizes and divides the plane in place, as Inductor fuses a reduction with its pointwise
  consumer.

PyTorch is BSD-3-Clause; `baseline/LICENSE` is its license.

## GPU risks (not yet run)

- Timing: the smallest performance case is estimated above 10 us (2.3 GFLOP of fp32 MFMA work
  on 121 output tiles); not measured.
- Numerics: the 124 dB threshold was calibrated with CPU emulations of float32 accumulation
  orders, not with the real fp32 MFMA rounding; the audit's baseline and control SQNRs should
  confirm it (an emulation of the baseline's arithmetic gives 132.7 dB).
- The baseline was checked for correctness only with the Triton CPU interpreter on small shapes
  (SQNR 129-135 dB there, with NumPy's float32 matmul in place of MFMA and a sequential
  combine of the Welford partials).
