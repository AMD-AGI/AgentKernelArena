# a16w8 block-scaled GEMM, N = 256, K = 7168

`out = x @ dequant(w)^T`: bfloat16 activations `x` [m, 7168] (not quantized) times float8_e4m3fn weights
`w` [256, 7168] with one float32 scale per 128x128 weight block (`w_scale` [2, 56]). Each K block of the
product is accumulated in float32, multiplied by the scale of its weight block and summed; the result is
rounded to bfloat16. The shape is the router-sized projection of a DeepSeek-style model (256 outputs over a
7168-wide hidden state), a small-N GEMM that is bound by reading the activations.

**Inputs.** `inputs.py` draws normal weights (std 0.02) and quantizes each 128x128 block with scale
`amax / 448`, so codes and scales are consistent. Activations are standard normal. The reference
dequantizes and multiplies in float64.

**Workloads.** The two old tasks (an AITER Triton extraction and a PyTorch-oracle FlyDSL task) both contained
N = 256, K = 7168 with M of 32 to 128; this is the scenario they share. M is the token count: small values
(decode-sized, below the 10 us floor) are correctness-only; performance cases run from 2048 to 204800 tokens
(100x). The 10 us floor needs about 2048 tokens (activation read 29 MB plus a launch floor).

**Baseline.** AITER's Triton `_gemm_a16w8_blockscale_kernel` (see `provenance` in `task.yaml`; `baseline/LICENSE`
is the AITER MIT license). Production selects a tile and split-K factor from a tuned table; split-K needs a
second reduce kernel and is only used upstream for M <= 128, so the baseline keeps the three rows without
split-K (M <= 256, M <= 512, otherwise). Every performance case uses the large-M row (128x128x128 tile,
8 warps).

**GPU audit risks.** The baseline is only 2 N-tiles wide, so small-M performance cases use few workgroups
and may sit close to the 10 us floor; the weight dtype is e4m3fn, which is the gfx950 format.
