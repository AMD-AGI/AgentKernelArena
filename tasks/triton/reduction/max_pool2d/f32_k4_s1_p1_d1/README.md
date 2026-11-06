# MaxPool2d, float32, kernel 4, stride 1, padding 1, dilation 1

`out[n, c, i, j]` is the maximum of the 4x4 window of `x[n, c]` starting at `(i - 1, j - 1)`,
taking only positions inside the image (padding is negative infinity). The input is NCHW float32
with 64 channels; batch, height and width vary, and the output is `(height - 1) x (width - 1)`.
The operator is KernelBench level 1 problem 42 (`Max_Pooling_2D`); it is memory bound.

Workloads: the original problem's input shapes (`16x32`, `32x64` and `8x16` planes at 256 and 512,
and `64x64x128x128`) rescaled to a fixed 64 channels, plus smaller and non-square maps. Maps whose
traffic is below the 10 us timing floor, and the boundary shapes, are correctness-only.

Bound: `exact` (the justified departure from the default bound). A maximum selects one input element, so the float64 reference rounded to float32 is
bit-identical to any correct implementation; no tolerance is needed or granted. Inputs are uniform in
[0, 1) like the original `torch.rand` inputs, so no NaN or infinity occurs.

Baseline: a Triton rewrite of the pointwise windowed-max kernel that PyTorch Inductor generates for
`F.max_pool2d` (one program per 1024 flattened outputs, 16 masked loads each, no indices output). It is
a standalone `@triton.jit` kernel without `torch._inductor` imports; provenance records the PyTorch
revision whose lowering it follows. The old task's baseline was the eager PyTorch module, which cannot
be vendored as a kernel.
