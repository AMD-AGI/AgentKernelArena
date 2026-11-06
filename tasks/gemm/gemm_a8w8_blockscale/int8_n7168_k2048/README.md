# a8w8 block-scaled GEMM, int8, N = 7168, K = 2048

`out = (a * a_scale) @ (b * b_scale)^T`: int8 activations `a` [m, 2048] with one float32 scale per 1x128
token group (`a_scale` [m, 16]) times int8 weights `b` [7168, 2048] with one float32 scale per 128x128 block
(`b_scale` [56, 16]). Each 128-wide K block of the integer product is accumulated exactly, multiplied by the
product of its activation and weight scales and added in float32; the result is rounded to bfloat16.
The shape is the shared-expert down projection of a DeepSeek-V3-style model (expert width 2048 back to a
7168-wide hidden state), the layer that vLLM's block-quantized int8 path serves with this kernel.

**Inputs.** `inputs.py` draws normal activations (std 1) and weights (std 0.02) and quantizes them with the
rule of vLLM's per-token-group int8 quantization: scale = amax / 127 over the group (1x128 for activations,
128x128 for weights), code = round(value / scale) clamped to [-128, 127]. Codes and scales are therefore
consistent, and the activation quantization (done upstream in production) is part of the input, not timed.
The reference dequantizes and multiplies in float64.

**Workloads.** M is the token count. The previous task for this kernel used toy shapes (M <= 512, N <= 512);
this scenario uses the production N and K. Decode-sized batches and batches under about 1024 tokens run
below the 10 us floor and are correctness-only; performance cases run from 1024 to 131072 tokens (128x).

**Baseline.** vLLM's Triton `_w8a8_block_int8_matmul` kernel with the default tile configuration of
`w8a8_block_int8_matmul` (64x128x128 tile, 4 warps, 3 stages, GROUP_SIZE_M 32); vLLM ships tuned tables only
for NVIDIA devices, so on MI355X this default is what the production function runs. The kernel text is
unchanged; the launcher writes into the preallocated output (see `provenance` in `task.yaml`). The tile
is a fixed 64x128 for every M, so it is a deliberately untuned starting point.

**GPU audit risks.** Performance at M = 1024 is estimated close to the 10 us floor only if the baseline
reaches peak; the untuned tile is far from it, so the floor should hold. `tl.dot` on int8 operands accumulates
in int32 and is converted to float32 per block, as upstream.
