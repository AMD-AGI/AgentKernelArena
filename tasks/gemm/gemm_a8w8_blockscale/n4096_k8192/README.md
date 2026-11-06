# a8w8 block-scaled GEMM, N = 4096, K = 8192

`out = dequant(a) @ dequant(b)^T`: float8_e4m3fn activations `a` [M, 8192] with one float32 scale per 1x128 group
(`a_scale` [M, 64]) times float8_e4m3fn weights `b` [4096, 8192] with one float32 scale per 128x128 block
(`b_scale` [32, 64]). Each K block of the product is accumulated in float32, multiplied by the activation and weight
scale of its block and summed; the result is rounded to bfloat16. This is the DeepSeek-style block-scaled FP8
linear layer (activations quantized per token in groups of 128, weights per 128x128 block) with a 4096-wide output
over an 8192-wide input.

**Inputs.** `inputs.py` plays the role of the producers. Activations are normal values with a per-(row, 128-column)
magnitude drawn over a 16x range, quantized to e4m3fn with scale `amax / 448`; weights are the same per 128x128 block.
The kernel-facing weight is preshuffled once with the AITER `shuffle_weight(layout=(16, 16))` permutation
(`to_kernel_layout`, outside timing, as production does at weight-load time), and `a_scale` is the column-major
payload of the activation-quantization kernel (a `[K/128, M]` buffer presented as `[M, K/128]`): activation
scales are produced by the upstream quantization kernel in that layout, so it is the kernel-facing input. The
reference receives the logical operands (natural weight order, `a_scale[row, group]`) and dequantizes and
multiplies in float64, chunked over rows.

**Workloads.** The original task (an AITER row with raw column-major activation scales, M = 1 to 4096) enumerated
power-of-two token counts; they are kept as the decode and prefill shapes. At M below about 32 the 33 MB weight
read (about 7.5 us) plus the launch floor sits at the 10 us line, so those sizes are correctness-only; performance
cases run from 32 to 8192 tokens (256x). Correctness cases add the boundary sizes 1, 3, 5, 17, 100 and 1000.

**Baseline.** AITER's Triton `gemm_a8w8_blockscale_preshuffle` (`_gemm_a8w8_blockscale_preshuffle_kernel`, see
`provenance` in `task.yaml`; `baseline/LICENSE` is the AITER MIT license). It replaces the CK / assembly
`gemm_a8w8_blockscale_bpreshuffle` of the original tasks, which cannot be vendored as source.
Upstream lowers `num_stages` from 3 to 2 on gfx950 for Triton releases before 3.7 (a compiler bug in the AMD buffer-ops
conversion); the baseline keeps that guard. The upstream kernel calls `remap_xcd(pid, GRID_MN)` without using the
result (a no-op); it is kept unchanged. The generic gfx950 preshuffle table is used because AITER has no N=4096, K=8192 specialization for gfx950.

**GPU audit risks.** Small-M performance cases (32, 128) are dominated by the weight read and may sit close to the
10 us floor; the largest case is compute bound (about 0.55 TFLOP). fp8 type is e4m3fn, the gfx950 format.
