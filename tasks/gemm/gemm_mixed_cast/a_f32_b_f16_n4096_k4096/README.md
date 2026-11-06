# Mixed-precision GEMM with an in-kernel cast, float32 x float16 -> float16, N = K = 4096

`out = float16(a) @ b`: the float32 activation `a` [m, 4096] is rounded to float16 inside the operator (the
operand is cast to the output dtype before the product, the point of triton issue 2523 and the rocmbench
`test_cast_matmul` test), multiplied with the float16 weight `b` [4096, 4096] (row-major, K x N), accumulated in
float32 and rounded to float16. The reference rounds `a` to float16 exactly as the operator does and then multiplies
in float64, so the cast is part of the checked semantics rather than a tolerance.

**Origin.** The old task enumerated every (A, B, C) dtype triple of {float16, float32, float64} x {float16, float32}
at two tiny shapes (128^3 and 1280x768x1024) with a loose `atol=0.3, rtol=0.01` check and a declared dot accumulator
equal to the output dtype. A schema-v3 task fixes one scenario, so this one takes A float32, B float16, C float16.
The dot accumulator is float32: a float16 accumulator over K = 4096 has no meaningful accuracy and is not a
production configuration. The operator has low production value of its own; the scenario is kept as the mixed-dtype
GEMM slot (cast fused into the load path, no separate conversion kernel).

**Inputs.** `a` standard normal, `b` normal with std 0.05 (so |out| stays around 3, well inside float16).

**Workloads.** The old shapes were far too small to time. Row counts here run from 256 to 32768 (128x) at the fixed
4096x4096 weight (32 MB). The estimated baseline time at M = 256 is about 12 us (8.6 GFLOP / 1e15 plus the 4 us
floor); smaller batches are correctness-only. The largest case reads 512 MB of float32 activations.

**Baseline.** The rocmbench `matmul_kernel` unchanged (grouped program ids, masked K tail, per-tile `.to` casts of
both operands, `tl.dot` with a float32 accumulator), wrapped in `run()` with a fixed 128x128x64 tile and GROUP_M 8
(the old harness used 16x16x32, which is far from a usable tile). `baseline/LICENSE` carries the MIT license text of
the rocmbench source.

**GPU audit risks.** Lowest performance case (M = 256) is near the 10 us floor; raise the minimum performance case if
the audit measures it below. The float32 operand reads make the kernel bandwidth-sensitive at large M; the bound is the
default baseline-relative bound.
