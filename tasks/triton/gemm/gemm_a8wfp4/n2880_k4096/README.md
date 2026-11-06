# a8wfp4 GEMM, N = 2880, K = 4096

`out = (a * a_scales) @ dequant(w)^T`: float8_e4m3fn activations `a` [m, 4096] with one float32 scale per token
(`a_scales` [m, 1]) times MXFP4 weights. `w` [2880, 2048] holds two e2m1 codes per byte (the even K index is the low
nibble) and `w_scales` [2880, 128] holds uint8 e8m0 scales (`2^(s - 127)`), one per 32 values of K. The block
products are accumulated in float32, the row scale is applied at the end and the result is rounded to bfloat16.
The shape is the attention output projection of a GPT-OSS-style model (2880 hidden, 4096 = 64 heads x 64).

**Reference.** `reference.py` dequantizes both operands (fp8 value times its row scale, e2m1 value times its block
scale) and multiplies in float64, row-chunked to bound memory. It is independent of the baseline and of the
quantizers in `inputs.py` (it consumes codes and scales only).

**Inputs.** `inputs.py` draws normal activations (std 1) and normal weights (std 0.05) and quantizes them: per-row
scale `amax / 448` and e4m3fn codes for the activations; for the weights one e8m0 scale per 32 values of K (group
maximum rounded to a power of two, minus the exponent of the largest e2m1 value), e2m1 codes rounded to nearest with
ties to the even code. Both operands are produced upstream of the operator in production, so quantization is part of
the input, not timed.

**Workloads.** The old FlyDSL task of this operator ran (M, N, K) of (128, 256, 7168), (256, 5120, 2880),
(128, 2880, 4096), (512, 2880, 4096) and (1024, 1024, 1024) with a PyTorch model as baseline. This scenario is its
N = 2880, K = 4096 shape with M extended to prefill token counts so that 100x of var-axis span is reachable.
The weight is 5.9 MB, so counts up to 512 (including the old task's 128 and 512) are estimated under or at the 10 us
floor (fp8 flops / 2e15 plus a 4 us floor is about 10 us at M = 512) and are correctness-only; performance cases run
from 1024 to 131072 tokens (128x). The largest case writes a 755 MB output; its float64 reference output is 3 GB.

**Baseline.** AITER's Triton `_gemm_a8wfp4_kernel` (`tl.dot_scaled` e4m3 x e2m1) with the launcher, `_get_splitk`, the
split-K reduce kernel and the generic gfx950 table inlined (see `provenance` in `task.yaml`; `baseline/LICENSE` is
the AITER MIT license). AITER has no N/K-specialized file for this shape. For M <= 128 the table asks for split-K 4,
which `_get_splitk` adapts to K = 4096 (two active splits of 2048 for BLOCK_SIZE_K 256) and the launcher disables when
M is below the M tile; partial sums are float32 and are allocated inside `run` (an allocation, allowed in
candidates). For M > 128 there is no split-K. The old task's baseline was a PyTorch model; the AITER Triton kernel
replaces it, so speedups are relative to the production kernel.

**GPU audit risks.** (1) Upstream carries a comment that the split-K path produced incorrect partial sums for small
M on gfx950 and works around it only for M below the M tile; the correctness-only cases M = 8 and 128 exercise
the split path and should be checked on MI355X first. (2) M = 512 is near the 10 us floor and is correctness-only,
M = 1024 (about 16 us estimated) should be verified to clear it. (3) `tl.dot_scaled` needs gfx950.
