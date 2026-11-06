# InnerProd, float32, 512 channels

`out[b, 0, h, w] = sum_c (feat_img[b, 0, c] * scale[c]) * feat_sound[b, c, h, w] + bias`: a batched
matrix-vector product (`torch.bmm([B, 1, C], [B, C, H*W])`) of the weighted image feature vector with every
pixel of a sound feature map, plus a scalar bias, as in the InnerProd head of a sound-source
separation network. Everything is float32 and `C = 512`; `batch`, `height` and `width` vary (the performance cases use width 32).

- **Merged sources.** `hip2hip/InnerProd` and `torch2hip/gpumode/11709_InnerProd` are the same operator
  (gpumode 11709, `fc_dim = 512`). The module also has `forward_nosum` and `forward_pixelwise` modes; only the
  `forward` mode is kept: `forward_nosum` (an elementwise broadcast product) and `forward_pixelwise` (an
  image-by-sound pixel product) are different operators.
- **Inputs.** `feat_img` and `feat_sound` are standard normal; `scale` is uniform in [0.5, 1.0] (the old
  harness used a 0.5 to 1.0 channel ramp) and `bias` a scalar tensor of shape `[1]` with a standard
  deviation of 0.5. The output is a sum of 512 products, so its standard deviation is about 17.
- **Baseline.** The provided HIP kernel of `hip2hip/InnerProd` (task-original, not an upstream production
  kernel), `forward` mode only: one thread per output pixel, a grid-stride loop, `scale` staged in shared
  memory, products of `feat_img * scale` formed in float32 and accumulated in double. The host-side
  `contiguous()` copy and `torch::empty` are gone: the kernel writes the preallocated output.
- **Comparison: `sqnr` with `min_db: 121.5` instead of the default `baseline_relative`.** The baseline
  accumulates in double, so its own error is only the final float32 rounding (about 146 dB SQNR); the
  element-wise default bound (twice that error plus one rounding) would reject every float32-accumulating
  kernel. The error of a 512-term float32 sum is accumulation rounding, which depends on the summation order:
  with one sequential accumulator its RMS is about `6.8u` of the output's RMS (`u = 2^-24`, absolute errors up
  to 7e-5), and on single large outputs it exceeds `16u` of the output, so no element-wise bound can tell a
  correct float32 kernel from the auditor's `shifted_16_ulp` control (every output scaled by `1 + 16u`).
  The earlier `tolerance` bound (`atol = 1e-4`, `rtol = 1e-5`, a relative slack of about `170u`) accepted
  that control. Whole-output SQNR separates them, because the shift is systematic while accumulation
  rounding averages out. CPU emulations of float32 accumulation orders on this task's input distributions,
  scored as the evaluator scores them (against the reference rounded to float32):

  | Implementation | SQNR |
  | --- | --- |
  | one float32 accumulator, sequential over the 512 channels (FMA or multiply-then-add, forward or reversed) | mean 127.6-128.0 dB; lowest 121.4 dB in 420,000 draws of a 64-element output (one draw below 121.5 dB, three below 122 dB), lowest 125.6 dB in 100,000 draws of 448 to 555 elements, at least 127.4 dB at 8,192 elements |
  | four interleaved accumulators, or blocks of four products (fp32 MFMA) | 133.4-133.7 dB |
  | split over the channels into 8 sequential chunks | 136.0 dB |
  | 64 lane accumulators and a tree combine | 140.5 dB |
  | the baseline's arithmetic (double accumulation) | 146.5 dB; lowest 141.6 dB at 64 elements |
  | `shifted_16_ulp` control | 120.40 dB mean; at most 120.9 dB at 64 elements, 120.55 dB at 448 elements and more, 120.44 dB at 8,192 elements and more |
  | operands rounded to 16 mantissa bits (a two-term bfloat16 split) | 107 dB |
  | TF32 operands | 71 dB |

  121.5 dB therefore accepts every correct float32 order (a sequential accumulator failed one in 420,000
  draws of a 64-element output and none of a larger one) and rejects the 16-ulp shift in every case,
  by at least 0.6 dB in the 64-element cases and 1 dB in the cases of 448 elements and more, where the
  shift's SQNR barely varies. Unwritten (NaN-poisoned) or non-finite elements always fail; the zero, stale,
  half-written and fp8-rounded controls fail by tens of dB, and so do a dropped channel (about 27 dB) and a
  missing bias (about 30 dB for a typical bias draw). Error spread over the output fails above an RMS of
  about 1.5e-5 (the old tolerance allowed about 2.7e-4 on every element); a single wrong element fails above
  about 1.2e-4 in the 64-element cases, 3e-4 in the 448- to 555-element cases and 0.02 in the largest case.
  The harness `dot_product` bound was not used because the evaluator provides no magnitude reference for it.
- **Workloads.** Feature maps of `B x H x 32` from `16 x 32 x 32` up to `256 x 256 x 32` pixels (`B * H` from
  512 to 65536, a 128x span; the largest `feat_sound` is 4.3 GB). The gpumode cases (batch 4 to 32, 16x16 and
  32x32 maps) move 4 to 67 MB and are partly below the 10 us floor; the performance cases start at 16 x 32
  rows (about 33 MB of `feat_sound`). The correctness-only cases cover the axis minimum, non-power-of-two and
  unaligned batch, height and width values (including the old suite's square 4x4, 16x16 and 64x64 maps), and
  the maximum height and width. Every correctness case has at least 64 output elements: whole-output SQNR is
  noisy on fewer, and a correct sequential accumulator fell below 121.5 dB in 0.34 % of 14-element draws and
  0.02 % of 32-element draws. The width-minimum case therefore uses batch 64 (`b64_h7_w1`, 448 elements) and
  the batch-and-height-minimum case the maximum width (`b1_h1_w64`).
- **Measured on MI355X.** The QA audit timed the baseline at 72 us for the smallest performance case
  (`b16_h32`) and 748 us for the largest (`b256_h256`).
- **Risk.** The 121.5 dB threshold was calibrated with CPU emulations of float32 accumulation, not on the GPU.
  The emulation rounds every float32 add and multiply as IEEE round-to-nearest does, and every FMA up to a
  negligible double rounding, which is what a HIP kernel's float arithmetic compiles to.
