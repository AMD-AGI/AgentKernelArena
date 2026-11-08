# conv2d, 3x3 stride 1 padding 1, NHWC bf16, 512 -> 512 channels, Winograd-domain filters

`out[n, p, q, o] = bias[o] + sum_{r,s,c} x[n, p+r-1, q+s-1, c] * g[o, c, r, s]` with zero padding of one pixel, bfloat16
channels-last input and output and float32 bias, computed in AITER by Winograd F(4x4, 3x3): an input transform
(6x6 patches of each 4x4 output tile), 36 batched GEMMs over the channels (V[36, T, C] x U[36, K, C]), and an output
transform, in three Triton kernels. T = batch x ceil(height / 4) x ceil(width / 4) tiles.

## Interface: the filters arrive in the Winograd domain

AITER computes the filter transform once per weight (`prepack_winograd_filter_f4x3`, cached) and the kernels consume
`U[6 i + j, o, c] = (G g[o, c] G^T)[i, j]`, rounded to bfloat16, with

    G = [[ 1/4,     0,    0  ],
         [-1/6,  -1/6, -1/6  ],
         [-1/6,   1/6, -1/6  ],
         [ 1/24,  1/12, 1/6  ],
         [ 1/24, -1/12, 1/6  ],
         [ 0,       0,    1  ]]

So `weight` has shape [36, 512, 512] and is produced by the task's layout transform from the logical OIHW filters
[512, 512, 3, 3]. The float64 reference uses the logical filters. An implementation may use the transformed filters
directly (as the baseline does) or recover a direct filter from them; the bfloat16 rounding of U is a property of the
operator's input that the baseline's error already contains.

## Scenario and sources

The layer is a 512 -> 512 3x3 convolution of the Stable Diffusion 3.5 / Flux-family VAE decoder (`AutoencoderKL`): the
resnet convolutions of the mid block and the first two up blocks, which are 512 channels wide.
- Stable Diffusion 3.5 Medium `vae/config.json` (Hugging Face): `block_out_channels` [128, 256, 512, 512]; the mid block
  and the first two up blocks use 512 channels (the decoder's up blocks use the reversed list [512, 512, 256, 128]). The
  Stable Diffusion v1.5 and SDXL VAEs have the same widths.
- The mid block and first up block run at the latent size, 1/8 of the image side (64x64 for a 512x512 image); the second
  up block runs at twice that.
- The same shapes are the upstream benchmark entries `C=512, K=512, R=3, pad=1` at 64x64, 128x128 and 256x256 of
  `stable-diffusion-3.5-medium` in `op_tests/op_benchmarks/triton/conv_shapes.json` of the pinned AITER commit.

Dispatch: with `layout="nhwc"` AITER's `_resolve_route` takes the Winograd kernels (`Route.WF4X3`) when
`_select_3x3_method` picks either Winograd variant: stride 1, dilation 1, C >= 512, K >= 512 and T >= 98. This task fixes
C = K = 512 and its axes keep T >= 100 (batch >= 1 and height, width >= 40), so the Winograd route is taken for every
case; below that tile count AITER would choose the direct kernel. (At T >= 392 AITER prefers a channel-blocked NCHW input
variant; with NHWC input it runs this same route.)

Activations are SiLU of a standard normal in bfloat16 (what GroupNorm + SiLU feed the convolution); filters follow
nn.Conv2d's default initialization (uniform in +-1/sqrt(512 x 9), bfloat16) and the bias is float32 uniform in that
bound. Trained weights have other statistics; the kernels are data-independent.

## Workloads

Var axes: batch 1-64, height and width 40-160 (the VAE latent of 320x320 to 1280x1280 images; 64 and 128 are the 512 and
1024 pixel cases). Performance cases run from one 64x64 map (4096 pixels) to 32 maps of 128x128 (524288 pixels), 128x in
batch x height x width. The pipeline writes V (bfloat16) and M (float32) workspaces, so it is memory-bound: estimated
about 85 MB of traffic and 5 us of GEMM for the smallest case (roughly 25 us with 4 us fixed cost) and 8.8 GB and 0.6 ms of
GEMM for the largest (about 2.6 ms). Correctness-only cases use heights and widths that are not multiples of 4 (ragged
last tile row and column), odd batches and a 100x75 map. The largest case allocates 3.6 GB of workspace and a 2.5 TFLOP
float64 reference.

## Baseline

AITER's Triton conv2d routed to its Winograd F(4x4, 3x3) pipeline with the gfx950 tables' single tiles (GEMM 64x64x64,
4 warps, 1 stage; transforms BLOCK_C = BLOCK_K = 64). `task.yaml` lists every file used and the modifications. The
workspaces V and M are allocated inside `run()` as the upstream launcher does (allocation only). The kernels take the tile
count, spatial sizes and channel counts as compile-time constants, so each distinct (batch, height, width) compiles
three kernels once.

Numerics: the input transform stores V in bfloat16 and U is bfloat16, so the baseline's error against float64 is larger
than a direct convolution's (about 32 dB SQNR measured on a CPU emulation of the pipeline for 512 input channels, above
the framework's 25 dB sanity floor; a direct bfloat16 convolution reaches about 53 dB). The default `baseline_relative`
bound therefore admits direct implementations (they are more accurate than the baseline) as well as Winograd ones; no
bound is changed.

All baseline files come from ROCm/aiter at commit `652f2d4b768239aefd515c929e84817681ff598d` (MIT; `baseline/LICENSE` is the repository license). Every vendored `.py` keeps its upstream header (`# SPDX-License-Identifier: MIT` / `# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.`) verbatim. `baseline/kernel_repr.py` (copied unchanged) and `baseline/activation.py` (an excerpt) come from upstream files that carry no header; the repository MIT license applies to them. `baseline/kernel.py` is new glue (its header repeats the launcher's upstream header). No third-party origin is recorded for any vendored file: none of them, nor the `conv/README.md` and `conv/DESIGN.md` of that directory, names an origin ("adapted from", "based on", URLs), and the first commits of the files are AITER-authored (the Triton conv2d introduction, the kernel-name helper, the Triton activation and MoE work). The Winograd F(4x4, 3x3) transforms are the textbook algorithm (Lavin and Gray); no code of another project is included.
