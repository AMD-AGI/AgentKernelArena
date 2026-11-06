# a8w8 block-scaled GEMM, N = 512, K = 4096

`out = dequant(a) @ dequant(b)^T`: float8_e4m3fn activations `a` [M, 4096] with one float32 scale per 1x128 group
(`a_scale` [M, 32]) times float8_e4m3fn weights `b` [512, 4096] with one float32 scale per 128x128 block
(`b_scale` [4, 32]). Each K block is accumulated in float32, multiplied by the activation and weight scale of its
block and summed; the result is rounded to bfloat16. A narrow projection (512 outputs over a 4096-wide hidden
state) of a block-scaled FP8 model.

**Inputs.** `inputs.py` plays the role of the producers. Activations are normal values with a per-(row, 128-column)
magnitude drawn over a 16x range, quantized to e4m3fn with scale `amax / 448`; weights are the same per 128x128 block.
The kernel-facing weight is preshuffled once with the AITER `shuffle_weight(layout=(16, 16))` permutation
(`to_kernel_layout`, outside timing, as production does at weight-load time), and `a_scale` is the column-major
payload of the activation-quantization kernel (a `[K/128, M]` buffer presented as `[M, K/128]`): activation
scales are produced by the upstream quantization kernel in that layout, so it is the kernel-facing input. The
reference receives the logical operands (natural weight order, `a_scale[row, group]`) and dequantizes and
multiplies in float64, chunked over rows.

**Workloads.** Two original tasks (AITER rows, M = 1 to 4096) differed only in how the activation scale was stored
(a column-major raw payload versus logical values relaid out per call); v3 keeps the production column-major
payload as the kernel-facing input and merges them (the logical-storage variant is not a separate kernel-facing
input). The weight is only 2 MB, so a call is bound by tile latency and launch: with the tuned 128x64 and 128x128
tiles an under-512-token call runs only a few dozen workgroups and sits at the 10 us line. Small batches (1 to 256
including every tuned-table M bound, plus 17, 100 and 1000) are correctness-only; performance cases run from 512 to
65536 tokens (128x), prefill chunks up to a large batched prefill of a serving engine.

**Baseline.** AITER's Triton `gemm_a8w8_blockscale_preshuffle` (`_gemm_a8w8_blockscale_preshuffle_kernel`, see
`provenance` in `task.yaml`; `baseline/LICENSE` is the AITER MIT license). It replaces the CK / assembly
`gemm_a8w8_blockscale_bpreshuffle` of the original tasks, which cannot be vendored as source.
Upstream lowers `num_stages` from 3 to 2 on gfx950 for Triton releases before 3.7 (a compiler bug in the AMD buffer-ops
conversion); the baseline keeps that guard. The upstream kernel calls `remap_xcd(pid, GRID_MN)` without using the
result (a no-op); it is kept unchanged. The generic gfx950 preshuffle table is used because AITER has no N=512, K=4096 specialization for
this operator.

**GPU audit risks.** The smallest performance case (512 tokens, 32 workgroups of the 128x64 tile) is estimated at
only 10-15 us and may measure under 10 us; if it does, demote it and rebalance the span. The 65536-token case needs
about 0.27 GB for the activation and a chunked float64 reference.
