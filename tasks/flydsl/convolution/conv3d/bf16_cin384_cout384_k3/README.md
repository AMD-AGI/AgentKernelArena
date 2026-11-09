# conv3d, bfloat16, 384 -> 384 channels, 3x3x3, causal-padded (FlyDSL baseline)

`out = conv3d(x, weight, bias)` for a video feature map `x` [1, 384, d, h, w] (NCDHW, bfloat16), a 3x3x3 filter with 384
output channels and a bias, stride 1, zero padding of one along height and width and none along depth, so `out` is
[1, 384, d - 2, h, w] (bfloat16, accumulated in float32). The depth axis is "pre-padded": the input already holds the
two causal frames (the feature cache of previously decoded frames, or zeros for the first chunk) in front of the new
frames, which is how a causal VAE convolution is evaluated. The weight is packed as `[384, 27 * 384]` with the 384 input
channels fastest and taps ordered (T, R, S); the bias is float32.

## Scenario and config citations

The residual-block convolutions of the Wan2.1 video VAE decoder, as implemented in SGLang at
`sgl-project/sglang@887c401e151e2d1b082d74ef8769a23eef5c465c` (the pinned checkout; the public Wan2.1 weights are not
part of it):

- `python/sglang/multimodal_gen/configs/models/vaes/wanvae.py`, `WanVAEArchConfig`: `base_dim` 96, `dim_mult` (1, 2, 4, 4),
  `z_dim` 16, `scale_factor_spatial` 8, `scale_factor_temporal` 4.
- `python/sglang/multimodal_gen/runtime/models/vaes/wanvae.py`, `WanDecoder3d`: `dims = [dim * u for u in
  [dim_mult[-1]] + dim_mult[::-1]]` = (384, 384, 384, 192, 96); the first block (and the mid block) runs 384 -> 384 at the
  latent resolution and the first resample doubles the spatial size; the second block's first convolution is 192 -> 384
  and its other residual convolutions are 384 -> 384 at twice the latent resolution. `WanResidualBlock` convolutions are
  `WanCausalConv3d(in_dim, out_dim, 3, padding=1)`; `WanCausalConv3d` pads depth by `2 * padding` in front only (from the
  feature cache of `CACHE_T = 2` frames when present) and height and width by `padding`.
- `python/sglang/multimodal_gen/configs/sample/wan.py`: default 832 x 480 (81 frames), 1280 x 720 supported; with
  `scale_factor_spatial` 8 the latent maps are 60 x 104 and 90 x 160, and 81 frames give (81 - 1) / 4 + 1 = 21 latent
  frames.
- `python/sglang/multimodal_gen/configs/models/vaes/base.py`: tiled decoding uses 256 x 256 spatial tiles
  (`tile_sample_min_height/width`), i.e. 32 x 32 latents. The Wan config sets `use_tiling` False, so this is the tiled
  mode, not the default.

Constant dims: N = 1, C = K = 384, filter 3x3x3. Var axes: `d` from 3 (one new latent frame plus the two causal frames,
the feature-cache decoding step) to 23 (the whole 21-latent clip decoded at once), `h` and `w` from 8 up to the 180 x 320
map of 1280 x 720 at twice the latent resolution. The temporal growth of later decoder stages (temporal upsampling) is
not modelled.

## Reference and inputs

`reference.py` is `torch.nn.functional.conv3d` on float64 copies of the inputs with the same padding; it is independent
of the baseline. `inputs.py` draws the activations from N(0, 1), the weight from N(0, 0.01^2) rounded to bfloat16 (the
output standard deviation is then about 1), and the bias from N(0, 0.1^2) as a full-precision float32 vector (not rounded to bfloat16): the kernel's bias argument is a
float32 pointer and adds it in fp32, and a checkpoint's bfloat16 bias is a special case of that domain. The reference uses
exactly these float32 values. The packed weight layout is produced by `to_kernel_layout` outside timing;
the packing was checked on CPU against an explicit im2col matrix product in float64 (taps ordered T, R, S, channel
fastest). Upstream `_prep_weight` is `w.permute(0, 2, 3, 4, 1).contiguous().reshape(k, -1)`, which is what the
transform does. The reference receives the logical `[K, C, 3, 3, 3]` weight.

## Baseline

The implicit-GEMM conv3d of `kernels/conv/conv3d_implicit.py` (ROCm/FlyDSL
`dd8373349e33638c8d158a9ea009b7b28598984a`, tag v0.3.2, Apache-2.0, header `# Copyright (c) 2025 FlyDSL Project
Contributors` kept): `compile_transpose_ncdhw_ndhwc` (NCDHW -> NDHWC activation transpose) and
`compile_conv3d_implicit` (4-stage pipelined `mfma_f32_16x16x32_bf16` implicit GEMM, 128 x 128 tile, 2 x 4 waves, fp32
accumulation, bias added in fp32, bf16 output written in NCDHW). `kernel.py` follows `_conv3d_impl`: the activation
transpose runs on every call (it is timed, as upstream), then the conv kernel with the default tile and workgroup swizzle
1. `buffer_atomic_add`, called by the kernel's split-K store, is copied from `kernels/common/mem_ops.py` (same
header) instead of imported. Third-party origin: none named in `conv3d_implicit.py`; its history starts with FlyDSL's own
commit "conv3d: parameterize tile size + add autotuner" (#820).

Departures from the upstream call path, all forced by the candidate rules (no host arithmetic, no dtype conversion in
`run()`): the weight packing and the float32 bias are inputs prepared by the layout transform; upstream's split-K
(fp32 atomic accumulation, then a host-side `y.to(bfloat16)`) is not used. Upstream enables split-K only when `npq =
d_out * h * w` is a multiple of 128, lies in [4096, 8192) and the 128 x 128 tile grid is under 192 programs; none of the
cases below falls in that window (checked with the upstream heuristic), but a shape in the window would run single-pass
here, not as upstream would.

### FlyDSL 0.3.2 adaptation

None: the vendored FlyDSL is the version of the evaluation image, and `conv3d_implicit.py` was written against it.

## Workloads

Cases are decoding steps of the VAE at the latent maps above: one new frame (depth 3) at 32 x 32 (tile), 60 x 104, 90 x
160, and at twice those maps 120 x 208 and 180 x 320; and a whole 21-latent clip (depth 23) at 60 x 104 and 90 x 160.
Estimated baseline time (dense bf16 flops 2 * 384 * 10368 * `npq` / 1.0e15 plus 4 us): 12 us for the smallest case (32 x
32, 1024 output pixels) and 2.4 ms for the largest (302400 pixels), 108x in `d * h * w` (the product of the var axes:
3072 .. 331200). Correctness-only cases: depth 3 and 8 x 8 (the axis minimum, estimated about 4.5 us), 4 x 17 x 19, 5 x 30
x 53, 3 x 45 x 80 (360p latent map) and 7 x 33 x 40 (odd and unaligned extents, which exercise the kernel's M and N tail
masks). The largest case needs a 254 MB input, a 232 MB bf16 output and about 1.9 GB for the float64 reference output
(and 2 GB for the float64 input copy).

## Kernel notes

The smallest performance case (1024 output pixels) fills only 24 of the 256 CUs
with 128 x 128 tiles and is estimated near 12 us from the roofline; the actual kernel is likely slower. The conv kernel first compiles per shape (each case has its own extents), so
the correctness phase precedes timing; compile time grows with the number of cases (12). The transpose scratch
(`n * d * h * w * c` bf16) is allocated in `run()` on each call. Weight rows are `27 * 384 = 10368` elements, so
`crs % 32 == 0` and the K loop has no masked tail.
