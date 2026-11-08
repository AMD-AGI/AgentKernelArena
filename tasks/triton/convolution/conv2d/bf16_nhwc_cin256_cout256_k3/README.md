# conv2d, 3x3 stride 1 padding 1, NHWC bf16, 256 -> 256 channels

`out[n, p, q, o] = bias[o] + sum_{r,s,c} x[n, p+r-1, q+s-1, c] * weight[o, r*3+s, c]` with zero padding of one pixel,
bfloat16 channels-last input and output, float32 accumulation and float32 bias. The kernel is AITER's direct implicit-GEMM
3x3 kernel (no Winograd), M = batch x height x width, N = 256, K = 9 x 256.

## Scenario and sources

The layer is a 3x3 convolution inside the 256-channel up block of the Stable Diffusion 3.5 / Flux-family VAE decoder
(`AutoencoderKL`): the second convolution of each resnet block there and the conv after nearest-neighbour upsampling.
- Stable Diffusion 3.5 Medium `vae/config.json` (Hugging Face): `block_out_channels` [128, 256, 512, 512]; the decoder's
  up blocks use the reversed list [512, 512, 256, 128], so the third up block runs 256 -> 256 channels:
  `in_channels` = `out_channels` = 256. The Stable Diffusion v1.5 and SDXL VAEs have the same widths.
- The block runs at half the decoded image's side (a 256x256 map for a 512x512 image).
- The same shape is the upstream benchmark entry `N=1, C=256, H=256, W=256, K=256, R=3, pad=1` of
  `stable-diffusion-3.5-medium` in `op_tests/op_benchmarks/triton/conv_shapes.json` of the pinned AITER commit.

Dispatch: with `layout="nhwc"` AITER's `_resolve_route` takes the NHWC 3x3 kernel unless `_select_3x3_method` picks
Winograd, which requires C >= 512 and K >= 512; at 256 channels it never does, for any batch or size. (The channel-blocked
NCHW kernel needs a host-side input repack and is not part of this scenario.)

Activations are SiLU of a standard normal in bfloat16 (what GroupNorm + SiLU feed the convolution); filters follow
nn.Conv2d's default initialization (uniform in +-1/sqrt(256 x 9), bfloat16) and the bias is float32 uniform in that bound.
Trained weights have other statistics; the kernel is data-independent.

## Workloads

Var axes: batch 1-16, height and width 8-512. Performance cases run from one 128x128 map (a 256x256 image, 16384 pixels)
to eight 512x512 maps (2.1M pixels), 128x in batch x height x width. The kernel is compute-bound (1.18 MFLOP per pixel):
estimate at 1.0e15 FLOP/s plus 4 us is about 23 us for the smallest and 2.5 ms for the largest case. The 64x64 map would
take under 10 us; it and the small, odd or ragged shapes (13x29, 17x31, 100x75, batch 3 and 5) are correctness-only. The
largest case's float64 reference is a 2.5 TFLOP convolution.

## Baseline

AITER's Triton conv2d (`aiter/ops/triton/conv/conv2d.py`, `layout="nhwc"`) routed to `_conv2d_3x3_nhwc_kernel` with the
gfx950 table's single tile (64x64x64, 4 warps, 1 stage). `task.yaml` lists every file used and the modifications. AITER
repacks the OIHW filters once per weight to [out_channels, 9, channels padded to 64] and caches the result; the layout
transform performs that repack (256 is a multiple of 64, so nothing is padded and the kernel-facing shape is
[256, 9, 256]). The spatial sizes are compile-time constants of the kernel, so each distinct (height, width) compiles once.

All baseline files come from ROCm/aiter at commit `652f2d4b768239aefd515c929e84817681ff598d` (MIT; `baseline/LICENSE` is the repository license). Every vendored `.py` keeps its upstream header (`# SPDX-License-Identifier: MIT` / `# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.`) verbatim. `baseline/kernel_repr.py` (copied unchanged) and `baseline/activation.py` (an excerpt) come from upstream files that carry no header; the repository MIT license applies to them. `baseline/kernel.py` is new glue (its header repeats the launcher's upstream header). No third-party origin is recorded for any vendored file: none of them, nor the `conv/README.md` and `conv/DESIGN.md` of that directory, names an origin ("adapted from", "based on", URLs), and the first commits of the files are AITER-authored (the Triton conv2d introduction, the kernel-name helper, the Triton activation and MoE work). No code of another project is included.

The baseline runs no PyTorch compute in `run()` and launches one Triton kernel per call.

Comparison: the default `baseline_relative` bound (no override).
