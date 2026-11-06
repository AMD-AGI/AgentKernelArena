# FP8 GEMM with per-token and per-channel scales, N=4096, K=4096, bfloat16 output

`out = (x @ w^T) * (x_scale * w_scale)` with float8_e4m3fn `x` `[m, 4096]` (one float32 scale per row,
`x_scale` `[m, 1]`) and a float8_e4m3fn `[4096, 4096]` weight `w` in the `nn.Linear` layout (one float32 scale
per output channel, `w_scale` `[1, 4096]`). The products are accumulated in float32, multiplied by the outer
product of the scales and rounded to bfloat16. There is no bias. The reference is the same expression in
float64 (float8 values and their products are exact in float64).

## Production origin and workloads

The shape is a square 4096-wide linear layer (hidden size 4096, for example an attention output projection), as served with row-wise FP8 quantization (the AITER `gemm_a8w8` /
`gemm_a8w8_per_token_scale` operator; both have identical math: per-token activation scale, per-channel
weight scale). The old tasks (an AITER Triton extraction, an SGLang FP8 Triton image task for MI300X and two FlyDSL HipKittens 4-wave / 8-wave ports, all including 4096x4096x4096) are merged into this one scenario; the gfx942 `e4m3fnuz` variant is not carried over because the task targets gfx950, where the type is `e4m3fn`.
Cases span `m = 1 .. 65536`: decode batches and chunks below 512 tokens are correctness-only,
`m = 512 .. 65536` are performance cases (128x span). Boundary correctness cases: `m = 1`, `m = 3` (not a power
of two) and `m = 17`, `100`, `1000` (not multiples of 16).

**Inputs.** `inputs.py` quantizes normally distributed activations and weights row-wise (scale = row amax / 448,
clamped, cast to float8_e4m3fn), as a quantizing producer would, so codes and scales are consistent. The
weight scale is generated as a `[1, n]` row like the AITER interface.

## Baseline

AITER's Triton `_gemm_a8w8_kernel` (MIT, so `baseline/LICENSE` is included) with its launcher and the
tuned configuration table from the generic gfx950 file `gfx950-GEMM-A8W8.json` (AITER has no N=4096, K=4096 specialization for gfx950: 32-column tiles up to M = 64, 64x64 up to 128, 128x64 up to 256, 256x64 up to 512 and 128x128x128 above). The kernel is unchanged; the launcher is reduced to the
no-bias path (`HAS_BIAS=False`) writing into the preallocated output, and the split-K reduction kernel is
dropped because every entry of the table uses `NUM_KSPLIT = 1`. The old FlyDSL 8-wave kernel is not used as
the baseline.

## Estimated timing

Performance cases start at `m = 512`, estimated at about 9 to 12 us on the roofline model; the real Triton
kernel is expected to be slower. Smaller shapes are correctness-only. Not measured on GPU.

## Risks for the GPU audit

The `m = 512` case is near the 10 us floor. The float32 accumulation order differs between candidates; the
`baseline_relative` bound absorbs it.
