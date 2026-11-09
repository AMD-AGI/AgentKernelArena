# conv2d, 7x7 stride 2 padding 3, NHWC bf16, 3 -> 64 channels (ResNet stem)

`out[n, p, q, o] = sum_{r,s,c} x[n, 2p+r-3, 2q+s-3, c] * g[o, c, r, s]` with zero padding of three pixels, bfloat16
channels-last input and output, float32 accumulation, no bias. The output is ((H - 1) // 2 + 1) x ((W - 1) // 2 + 1). This
is AITER's general implicit-GEMM kernel (any kernel that is not 1x1 or 3x3): M = batch x output pixels, N = 64 and a
reduction of 3 x 7 x 7 = 147 taps, padded to 192.

## Scenario and sources

The layer is the stem `conv1` of ResNet-50 (torchvision `resnet50`: `nn.Conv2d(3, 64, kernel_size=7, stride=2,
padding=3, bias=False)`, followed by batch norm, which is why it has no bias), so `in_channels` 3 (RGB) and
`out_channels` 64 are constants. The same shape is the first entry of `resnet50` in
`op_tests/op_benchmarks/triton/conv_shapes.json` of the pinned AITER commit (N=1, C=3, H=W=224, K=64, R=S=7, stride 2,
pad 3). The upstream router sends it to the general kernel (`Route.GENERAL`) for every image size, and `layout="nhwc"`
is the channels-last format AMD training and inference stacks use.

Images are standard normals (standardized pixel values); filters follow torchvision's ResNet initialization
(`kaiming_normal_(mode="fan_out", nonlinearity="relu")`, std sqrt(2 / (64 x 49)), bfloat16). Trained stem filters have
other statistics; the kernel is data-independent. Real images are spatially smooth, which the random data is not; this
changes no control flow of the kernel.

## Workloads

Var axes: batch 1-256, height and width 16-1344. Ten performance cases, from 0.20M to 20.4M input pixels (102x in
batch x height x width). Times are the cold-cache CUDA-graph medians the baseline measured on MI355X where a case was
measured, otherwise the linear fit t = 13 us + 38.6 us per million input pixels to those measurements; memory is
the measured candidate-correctness rate described below.

| Case (batch x H x W) | Pixels | Output elements | Baseline time | Peak memory | Source |
| --- | --- | --- | --- | --- | --- |
| 4 x 224 x 224 | 0.20M | 3.2e6 | about 21 us (fit) | 0.5 GB | four ImageNet images |
| 8 x 224 x 224 | 0.40M | 6.4e6 | about 28 us (fit) | 1.0 GB | small inference batch |
| 16 x 224 x 224 | 0.80M | 1.3e7 | about 44 us (fit) | 1.9 GB | inference batch |
| 32 x 224 x 224 | 1.6M | 2.6e7 | 72.5 us | 3.8 GB | inference batch |
| 64 x 224 x 224 | 3.2M | 5.1e7 | 133 us | 7.6 GB | inference batch |
| 256 x 224 x 224 | 12.8M | 2.1e8 | 498 us | 30 GB | ResNet-50 per-GPU training batch |
| 32 x 512 x 512 | 8.4M | 1.3e8 | 351 us | 20 GB | high-resolution classification or segmentation |
| 64 x 512 x 512 | 16.8M | 2.7e8 | about 660 us (fit) | 40 GB | larger batch of the same |
| 16 x 832 x 1024 | 13.6M | 2.2e8 | about 540 us (fit) | 32 GB | detection-sized images (an 800x1333 resize padded to a multiple of 32, cropped) |
| 24 x 832 x 1024 | 20.4M | 3.3e8 | about 800 us (fit) | 48.5 GB | the largest case |

(The output-element column counts batch x ceil(H/2) x ceil(W/2) x 64.)

The stem moves about 38 bytes per input pixel and 4.7 kFLOP per input pixel, so it is bandwidth-bound in principle, but
the measured times follow about 13 us + 38.6 us per million pixels, well above the bandwidth roofline of about 8.4 us per
million: the 3-channel NHWC gathers and per-element index decoding of the general kernel dominate. Because one stem layer
needs a very large amount of work to span 100x, the largest cases are detection-sized batches, an offline-throughput
scenario rather than a latency one. A single 224x224 image (about 15 us by the fit, but a one-image case) and the odd or
ragged sizes (17x33, 31x47, 100x75, 223x225, batch 3, 5, 7) are correctness-only.

Memory. The baseline_relative check in candidate correctness keeps, besides the float64 reference of the output, the
float64 temporaries of the comparison and the baseline outputs. The measured rate (on a 288 GB MI355X, 128 x 832 x 1024 needs
at least 259 GB: 207 GB allocated plus a 52 GB request) is about 2.02 GB per 832 x 1024 image,
about 2375 bytes per input pixel or about 148 bytes per output element (about 18.5 float64 copies of the output). The
peak of each case is that rate times its pixels, as in the table (24 x 832 x 1024 = 20.4M pixels x 2375 B = 48.5 GB). A
simple count of 12 float64 copies undercounts it. The kernel uses 32-bit offsets (limit 2.1e9 elements); the largest
output (3.3e8 elements) uses 15% of that.

## Baseline

AITER's Triton conv2d routed to `_conv2d_general_kernel` with the gfx950 table's single tile (64x64x64, 4 warps,
1 stage). `task.yaml` lists every file used and the modifications. The filters arrive in the layout of AITER's cached
`prepack_oihw_to_kmajor`: [64, 147] in (c, r, s) order, zero-padded to [64, 192], produced by the layout transform from
the logical OIHW filters [64, 3, 7, 7]. Spatial sizes are compile-time constants of the kernel, so each distinct
(height, width) compiles once.

All baseline files come from ROCm/aiter at commit `652f2d4b768239aefd515c929e84817681ff598d` (MIT; `baseline/LICENSE` is the repository license). Every vendored `.py` keeps its upstream header (`# SPDX-License-Identifier: MIT` / `# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.`) verbatim. `baseline/kernel_repr.py` (copied unchanged) and `baseline/activation.py` (an excerpt) come from upstream files that carry no header; the repository MIT license applies to them. `baseline/kernel.py` is new glue (its header repeats the launcher's upstream header). No third-party origin is recorded for any vendored file: none of them, nor the `conv/README.md` and `conv/DESIGN.md` of that directory, names an origin ("adapted from", "based on", URLs), and the first commits of the files are AITER-authored (the Triton conv2d introduction, the kernel-name helper, the Triton activation and MoE work). No code of another project is included.

The baseline runs no PyTorch compute in `run()` and launches one Triton kernel per call.

Comparison: the default `baseline_relative` bound (no override).
