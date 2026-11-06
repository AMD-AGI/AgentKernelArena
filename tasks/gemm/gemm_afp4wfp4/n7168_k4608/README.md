# afp4wfp4 GEMM, N = 7168, K = 4608

`out = dequant(a) @ dequant(w)^T`: MXFP4 activations `a` and weights `w`, both stored as two e2m1 codes per byte
(the even K index is the low nibble) with uint8 e8m0 scales (`2^(s - 127)`), one per 32 values of K. `a` is
[m, 2304] bytes with `a_scales` [m, 144]; `w` is [7168, 2304] bytes with `w_scales` [7168, 144]. The block products are
accumulated in float32 and the result is rounded to bfloat16. The shape is the N = 7168, K = 4608 projection
listed by both old tasks of this operator (`gemm_afp4wfp4` and `gemm_a4w4`, which AITER serves with an assembly
kernel but which computes the same math; they are merged here).

**Reference.** `reference.py` dequantizes both operands from codes and scales and multiplies in float64, row-chunked
to bound memory. It is independent of the baseline and of the quantizer in `inputs.py`.

**Inputs.** `inputs.py` draws normal activations (std 1) and weights (std 0.05) and quantizes them like AITER's
dynamic MXFP4 quantization: the scale of a 1x32 group is the group maximum rounded to a power of two minus the
exponent of the largest e2m1 value, and codes round to nearest with ties to the even code, saturating at 6. In
production the activation quantization is a separate upstream kernel, so it is part of the input, not timed.

**Workloads.** The old tasks ran (M, N, K) of (192, 7168, 4608) and neighbouring shapes with a PyTorch model or
AITER as baseline. M is extended to prefill token counts so that 100x of var-axis span is reachable. The weight is
16.5 MB; estimated baseline time at M = 256 is fp8-rate flops (2e15, conservative for fp4) plus a 4 us floor, about
12.5 us, so counts up to 192 (decode, and the old task's 128 and 192) are correctness-only and performance cases run
from 256 to 65536 tokens (256x). The largest case writes a 940 MB output; its float64 reference output is 3.8 GB.

**Baseline.** AITER's Triton `_gemm_afp4wfp4_kernel` (`tl.dot_scaled` e2m1 x e2m1) with the launcher, `get_splitk`,
the split-K reduce kernel and the generic gfx950 table (`configs/gfx950/triton/gemm/GEMM-AFP4WFP4.json`) inlined (see
`provenance` in `task.yaml`; `baseline/LICENSE` is the AITER MIT license). AITER has no N/K-specialized file for
this shape (the file name uses the unpacked K, 4608). For M <= 8 the table asks for split-K 4, which `get_splitk`
turns into 3 splits of 1536 for K = 4608; partial sums are float32 and are allocated inside `run` (an allocation,
allowed in candidates), then added by the reduce kernel. For larger M there is no split-K. The AITER preshuffled
weight variant is not used; the plain layout is the interface. The old tasks' baselines were a PyTorch model and
AITER's assembly path; the AITER Triton kernel replaces them, so speedups are relative to the Triton kernel.

**GPU audit risks.** M = 256 is estimated close to the 10 us floor (the Triton kernel at that M uses 128x64 tiles
of K = 512 steps, so it should be above it, but this was estimated, not measured). The split-K path (M <= 8) is
correctness-only here (M = 1, 3, 8). `tl.dot_scaled` needs gfx950.
