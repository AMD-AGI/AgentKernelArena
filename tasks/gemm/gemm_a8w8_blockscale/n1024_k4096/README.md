# a8w8 block-scaled GEMM, fp8, N = 1024, K = 4096

`out = (a * a_scale) @ (b * b_scale)^T`: float8_e4m3fn activations `a` [m, 4096] with one float32 scale per
1x128 token group, times float8_e4m3fn weights of 1024 output channels with one float32 scale per 128x128
block (`b_scale` [8, 32]). Each 128-wide K block of the product is scaled by its activation and weight
scales and added in float32; the result is rounded to bfloat16. This is a projection of a large mixture-of-experts
model served with block-scaled fp8 (activation quantization runs in an upstream kernel).

**Storage.** The kernel-facing operands are the production ones of AITER's preshuffled block-scale GEMM,
produced by `inputs.py` outside timing: the weight is the (16, 16) preshuffled payload, shape
[n // 16, k * 16] (weight preparation done once at load time), and the activation scales are stored
transposed, `a_scale` [k // 128, m] (the quantization kernel writes them column-major). The reference sees the
logical operands (unshuffled [n, k] weight, [m, k // 128] scales), so a candidate must decode the layout. The two
previous tasks (raw and logical scale storage, CK/assembly baseline) differed only in how the scale matrix was
stored; this task keeps the column-major payload, which is what the kernel is called with in production.

**Inputs.** `inputs.py` draws normal activations (std 1) and weights (std 0.02) and quantizes them with
scale = amax / 448 per group (1x128 and 128x128) and e4m3fn codes, so codes and scales are consistent. The
reference dequantizes and multiplies in float64.

**Workloads.** M is the token count; the previous tasks covered M from 1 to 4096. Batches below about 2048 tokens
run under the 10 us floor and are correctness-only; performance cases run from 2048 to 204800 tokens (100x).

**Baseline.** AITER's Triton `_gemm_a8w8_blockscale_preshuffle_kernel` with its split-K reduce kernel and
tuned-config dispatch (see `provenance` in `task.yaml`; `baseline/LICENSE` is the AITER MIT license). Production
runs a CK/assembly function (`gemm_a8w8_blockscale_bpreshuffle`) on this shape, which cannot be vendored as source;
AITER's own Triton preshuffle variant, which consumes the same payload, replaces it, so the baseline may be slower than
production. The gfx950 file has no row specific to N = 1024, K = 4096, so the baseline uses the generic preshuffle table (`gfx950-GEMM-A8W8_BLOCKSCALE_PRESHUFFLED.json`); every performance case uses its large-M row (128x128x128 tile, 8 warps, no split-K), and the smallest correctness cases use its single-split rows. Three-stage configurations are clamped to two stages, as upstream does for Triton older than 3.7
on gfx950 (applied unconditionally here).

**GPU audit risks.** At M = 2048 the work (17 GFLOP of fp8) is only about 9 us at peak, so the first performance cases are estimated close to the 10 us floor; the baseline is unlikely to reach peak, but check the measured times.
