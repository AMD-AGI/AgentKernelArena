# Conv3d, channel softmax and two max pools, float32, 3 to 16 channels

**Operator.** KernelBench level 2 problem 6 (`Conv3d_Softmax_MaxPool_MaxPool`), the former
`torch2hip/kernelbench/level2/l2n6_Conv3d_Softmax_MaxPool_MaxPool` task:

```python
y = F.conv3d(x, weight, bias)       # 3 -> 16 channels, 3x3x3, stride 1, no padding
y = torch.softmax(y, dim=1)         # over the 16 output channels
y = F.max_pool3d(F.max_pool3d(y, 2), 2)
```

Everything is float32 in NCDHW layout. The output is
`[batch, 16, (depth - 2) // 4, (height - 2) // 4, (width - 2) // 4]`. Two floor-mode 2x2x2 stride-2
pools select exactly the maximum of each 4x4x4 stride-4 window, and trailing convolution rows that
fill no window are dropped.

**Scenario.** float32; 3 input channels, 16 output channels, a 3x3x3 kernel with bias and 16 input
frames are constant. `batch`, `height` and `width` vary. `weight` and `bias` are drawn uniformly
from +-1/9, the `nn.Conv3d` default initialization range for a fan-in of 81. `x` is uniform in
[0, 1), as KernelBench draws it.

**Workloads.** The KernelBench batches at its 16x32x32 clips (batch 32, 64 and the default 128),
C3D clips (16x112x112 crops at batch 16 and 64, full 16x128x171 frames at batch 32) and 16x224x224
clips (the VideoMAE and X3D-M input size) at batch 1, 8, 32 and 128: ten performance cases
spanning 196x in `batch * height * width`. KernelBench's batch 16 and 16x16 cases and a single
112x112 clip are correctness-only. At an MFMA float32 peak of 157 TFLOPS and 4.5 TB/s, the
estimated baseline time is 13 to 14 us for them, too close to the 10 us floor. Unaligned and
minimum-size frames (6x6, 37x45, 34x33, 18x66) cover partial windows and tile tails.

**Reference.** float64 PyTorch (`F.conv3d`, `torch.softmax`, two `F.max_pool3d`), computed in batch
chunks to bound memory.

**Bound.** `sqnr` with `min_db: 130` departs from the default `baseline_relative` bound. The
result is a float32 81-term convolution followed by a softmax, so a correct candidate that
accumulates in another order or uses another `exp` has rounding errors of about the same size as
the baseline's, but independent of them. The default bound compares each element with the
baseline's error at that element, or with its RMS over a pooled row of 1 to 55 elements. CPU
emulation measured how far correct float32 variants exceed that bound: PyTorch's own float32
`conv3d`, tap-by-tap accumulation in forward and reverse order, softmax through `exp2`, and softmax
without max subtraction. Under the default `k = 2`, 1 to 5 % of their elements fail. Even
`k = 8` with no allowed violations rejects some of them: on 10^4 to 3x10^5 elements, their peak
error reaches 9.5 times the bound's baseline term. Their SQNR, computed as the evaluator does,
is 139 to 142 dB, and the emulated baseline's is 141 to 143 dB. The least accurate correct
formulation tried, a log-sum-exp softmax `exp(z - logsumexp(z))`, reaches 136 dB. A 130 dB
bound admits an RMS relative error of about 5 units of float32 rounding (2^-24). It rejects
every control: the 16-ulp scaled output (120 dB), float8 rounding (31 to 33 dB), stale outputs
(about 20 dB), half-written or zero outputs and NaN. It also rejects TF32-precision convolution
inputs (80 to 87 dB), including on a single batch item of a large case.

**Baseline.** No production Triton kernel exists for this module, and Inductor's default choice
for the convolution is the MIOpen library call. The baseline is a Triton rewrite modeled on
Inductor's Triton convolution template (`conv3d_template` in `torch/_inductor/kernel/conv.py`,
PyTorch v2.15.0; `baseline/LICENSE` is the PyTorch BSD-3-Clause license). It uses two kernels:

1. An implicit GEMM over 128 output positions x 16 channels with IEEE float32 `tl.dot`. It reduces
   over the flattened (in_channel, kd, kh, kw) axis of the NCDHW input, in 32-wide blocks, rather
   than over per-tap channel blocks, because the input has only 3 channels. Bias and the channel
   softmax are applied in the epilogue, and the probabilities are written to a float32
   intermediate.
2. One 4x4x4 max-pool kernel over that intermediate.

A fused kernel that convolves only the positions inside pooling windows and never writes the
intermediate is left to candidates.

**GPU risks (not yet run).**

- The kernels have been emulated on the CPU only, which checked their index math against the
  reference. The `tl.dot` float32 IEEE path on gfx950 and the compile time of the 64-load
  unrolled pool kernel are unverified.
- `b32_h32_w32` is estimated at about 25 us, and the other correctness-only cases are estimated
  below 15 us.
- The largest case, `b128_h224_w224`, allocates a 5.6 GB intermediate. Its input is 1.2 GB, held
  in several copies by the evaluator.
- The float64 reference uses PyTorch's native GPU `conv3d` path, because MIOpen has no float64
  kernels.
