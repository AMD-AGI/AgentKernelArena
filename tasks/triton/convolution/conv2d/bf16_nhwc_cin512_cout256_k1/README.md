# conv2d, 1x1, NHWC bf16, 512 -> 256 channels

Pointwise convolution `out[n, h, w, o] = bias[o] + sum_c x[n, h, w, c] * weight[o, c]` of a bfloat16 channels-last
feature map, float32 accumulation, bfloat16 output. It is a GEMM with M = batch x height x width pixels, N = 256 and
K = 512.

## Scenario and sources

The layer is the `conv_shortcut` of the first resnet block in the 256-channel up block of the Stable Diffusion 3.5 /
Flux-family VAE decoder (an `AutoencoderKL` decoder in diffusers; the 1x1 shortcut exists because the block changes
from 512 to 256 channels).
- Stable Diffusion 3.5 Medium `vae/config.json` (Hugging Face): `block_out_channels` [128, 256, 512, 512]. The decoder's
  up blocks use the reversed list [512, 512, 256, 128], so the third up block takes 512 channels in and 256 out:
  `in_channels` 512 and `out_channels` 256 are constants of this scenario. The Stable Diffusion v1.5 and SDXL VAEs have
  the same channel widths (only `latent_channels` differs, which this layer does not see).
- The decoder reaches this block at half the decoded image's side (latent side x 4): a 256x256 map for a 512x512 image.
- The same shape is the upstream benchmark entry `N=1, C=512, H=256, W=256, K=256, R=1` of `stable-diffusion-3.5-medium`
  in `op_tests/op_benchmarks/triton/conv_shapes.json` of the pinned AITER commit.

Activations are bfloat16 standard normals (the residual stream feeding the shortcut); the filters follow nn.Conv2d's
default initialization (uniform in +-1/sqrt(512), here in bfloat16) and the bias is float32 uniform in the same bound.
Trained weights have other statistics; the kernel is data-independent, so the distributions only need to be realistic
in scale and non-degenerate.

## Workloads

Batch, height and width are the var axes: batch 1-16 images, feature maps 8-512 on a side (decoded images up to
1024x1024). Performance cases run from 25600 pixels (a 320x320 image) to 3.1M pixels (twelve 1024x1024 images), 123x in
batch x height x width. Roofline estimate for the baseline (bytes moved / 4.5e12 B/s, 1.5 KB per pixel, plus 4 us):
about 13 us for the smallest and 1.4 ms for the largest performance case. The 64x64 and 128x128 single-image maps
would take under 10 us and are correctness-only; small odd shapes (13x29, 17x31, 33x65, 100x75) and batch 3, 5, 7 cover
ragged M tiles. Largest case: x holds 1.6e9 elements (the kernel uses 32-bit offsets, whose limit is 2.1e9); its float64
reference needs about 13 GB for x and 6 GB for the output.

## Baseline

AITER's Triton conv2d (`aiter/ops/triton/conv/conv2d.py`, `layout="nhwc"`), which routes every 1x1, dilation-1
convolution to `_conv2d_1x1_kernel` with the gfx950 table's single tile (64x64x64, 4 warps, 1 stage). `task.yaml`
lists every file used and the modifications. Production prepacks nothing for 1x1 beyond a squeeze to
[out_channels, in_channels]; the layout transform does that squeeze (logical weight is OIHW [256, 512, 1, 1]). The
kernel's spatial dimensions are compile-time constants, so each distinct (height, width) compiles once.

All baseline files come from ROCm/aiter at commit `652f2d4b768239aefd515c929e84817681ff598d` (MIT; `baseline/LICENSE` is the repository license). Every vendored `.py` keeps its upstream header (`# SPDX-License-Identifier: MIT` / `# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.`) verbatim. `baseline/kernel_repr.py` (copied unchanged) and `baseline/activation.py` (an excerpt) come from upstream files that carry no header; the repository MIT license applies to them. `baseline/kernel.py` is new glue (its header repeats the launcher's upstream header). No third-party origin is recorded for any vendored file: none of them, nor the `conv/README.md` and `conv/DESIGN.md` of that directory, names an origin ("adapted from", "based on", URLs), and the first commits of the files are AITER-authored (the Triton conv2d introduction, the kernel-name helper, the Triton activation and MoE work). No code of another project is included.

The baseline runs no PyTorch compute in `run()` (views only) and launches one Triton kernel per call.

Comparison: the default `baseline_relative` bound (no override).
