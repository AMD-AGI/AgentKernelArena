# MXFP8 x FP8 scaled GEMM, N = 2048, K = 7168

`out = (a * a_scale) @ (w * w_scale)^T`: float8_e4m3fn activations `a` [m, 7168] with one e8m0 scale byte per 1x32 group
along K (`a_scale` [m, 224], microscaling MXFP8), float8_e4m3fn weights `w` [2048, 7168] with one e8m0 scale byte per
128x128 block (`w_scale` [16, 56]). An e8m0 byte `b` encodes `2^(b - 127)`. The product is accumulated in float32 and
rounded to bfloat16. The 128x128 weight scale is broadcast to 1x32 granularity inside the kernel (`tl.dot_scaled`).
N = 2048, K = 7168 is the DeepSeek-V3 hidden size (7168) projection shape.

**Inputs.** `inputs.py` draws standard-normal values rounded to float8_e4m3fn and e8m0 scale bytes in 125..129
(2^-2 .. 2^2), as the old tasks and the AITER operator test do. Codes and scales are independent random draws; the
reference dequantizes exactly what the kernel sees. The scaled GEMM consumes operands that were quantized upstream, so no
quantization is part of the operator.

**Reference.** Dequantizes both operands in float64 (value times decoded e8m0 scale) and multiplies in float64, in row
chunks of 8192 to bound memory (the largest case needs about 4 GB per chunk intermediate).

**Workloads.** From the old tasks (same operator in two of them; shared shape M=128, N=2048, K=7168), with M widened to
prefill token counts. The weight is 14.7 MB, so the estimated baseline time stays under 10 us (fp8 flops / 2e15 plus a
4 us floor) up to M of about 256; M <= 256 (including the decode-sized 128) are correctness-only. Performance cases run
from 512 to 65536 tokens (128x).

**Baseline.** AITER's Triton `_gemm_afp8wfp8_kernel` (non-split-K, one static 128x128x128 tile config, group size 8,
4 warps, 2 stages) from the old triton2flydsl task, wrapped in `run()`; see `provenance` in `task.yaml`.
`baseline/LICENSE` is the AITER MIT license. Only the single-tile static config is vendored; AITER's tuned-config
JSON lookup was already removed in the old task and is not part of this baseline. The one deviation from the AITER
launcher is `stride_ck` passed as `out.stride(0)` instead of 0 (unused without split-K). `tl.dot_scaled` exists only on
gfx950, hence `required_arch: [gfx950]`.

**GPU audit risks.** The lowest performance case (M = 512, estimated about 11 us) is close to the 10 us floor; if the
audit measures it below, raise the smallest performance case and keep the 100x span (raise the axis maximum
if needed). The baseline is Triton, so it also satisfies the Triton candidate rules (`run()` does only views,
`a.view(uint8)` and the weight transpose view). Bitwise differences from the reference come only from fp32 accumulation order.
