# GEMM, float32, n=4096 k=4096

Operator: `out = a @ b` (`torch.matmul`) for `a` [m, 4096] and `b` [4096, 4096], both row-major float32,
output float32 [m, 4096]. Products and accumulation are IEEE float32: TF32, bf16x3 or other
reduced-precision inputs are a different operator (PyTorch keeps float32 matmul precision at `"ieee"` by
default).

**Origin.** The former float32 matmul tasks, which are all this operator: KernelBench level 1 problems 1
(square matrix multiplication), 2 (standard matrix multiplication) and 8 (matmul with irregular shapes) from
`torch2hip/kernelbench/level1`, and the HIP tasks `hip2hip/matrix_multiplication` (the rocm-examples
HIP-Basic tiled matrix multiplication) and `hip2hip/SimpleMatmulModule` (`a @ (b + b)` on shapes up to 128 x
128). Their cases scaled m, n and k together, from 16 x 32 x 48 up to 8205 x 2949 x 5921. A scenario fixes
the inner sizes, so this task takes n = k = 4096 (the 4096^3 product appears in both KernelBench problem 1
and problem 2) and varies only m, the rows of `a`. The `(b + b)` prescale of `SimpleMatmulModule` and its
sub-microsecond shapes are not part of the scenario.

**Workloads.** Eight timed row counts from 64 to 8192 (128x span; 2.1 to 275 GFLOP). Even the smallest reads
all 64 MB of `b` from HBM after the cache flush, so it stays above the 10 us floor however fast the matrix
units are. Correctness-only boundary shapes: 16 (the axis minimum, the rocm-examples control shape row
count), 48 (not a power of two), 100 (not a multiple of 16), and the irregular KernelBench row counts 4099
and 8205, which exercise the partial last row tile.

**Inputs: uniform on [0.5, 1.5), not KernelBench's [0, 1).** KernelBench draws `torch.rand`, uniform on [0,
1). With those inputs a correct k-sequential FMA loop, the order of the rocm-examples kernel and of any
register-blocked SGEMM on the vector units (which MI355X runs at the same float32 rate as its matrix units),
fails the comparison's bias test on every large case. The product of two [0, 1) numbers has a density that
grows without bound near zero (-ln p); among the addends no larger than the accumulator's ulp, the tiny ones
that round down outnumber those that round up, so the loop's mean error is -0.54% of its RMS. At `m8192` (33.5
M elements) that is 6x the test's allowance of 5/sqrt(n) = 8.6e-4, and the fallback (k times the baseline's
own mean error) does not cover it, because the modelled baseline, whose 4-term steps rarely add tiny values,
is unbiased (z = +1.1e-4). Two-product pairs are biased the other way (+0.38%). Shifting both inputs to [0.5,
1.5) keeps every product at least 0.25, far above the accumulator's ulp: in a simulation of the same size the
FMA loop's z is -2.2e-4, and FMA'd pairs, 2-way split-K and the CPU float32 matmul stay within 4e-4 (16.8 M
elements), all passing. The outputs stay positive (about 4096), which is what lets the bias test reject a
scaled result (below).

The axis starts at m = 16, not 1: a one-row product is a matrix-vector product, a different kernel regime,
and its 4096-element output is too small for the violation allowance below (one element out of bound is
already 2.4e-4 of it), so a correct k-sequential FMA kernel would fail it in about one call in nine.

**Reference.** `torch.matmul` in float64.

**Comparison: `baseline_relative` with `k: 8`, `max_violation_fraction: 2.0e-4` (an override).** The output
is float32 and the baseline's error is the rounding of its 4096-deep float32 accumulation (an RMS relative
error of about 7 u, u = 2^-24 being the float32 unit roundoff; about 127.6 dB SQNR in simulation), not one
output rounding. Any other legitimate summation order has an independent error of similar size, and the
default bound (`k = 2`, no violations) rejects it. A CPU simulation against the float64 reference (the
task's [0.5, 1.5) inputs, k = 4096, 262,144 to 786,432 output elements; the baseline modelled as the MFMA
16x16x4 order, four-term dots added to a float32 accumulator in k order, which is what the Triton dot
compiles to: Triton folds `acc += tl.dot(a, b)` into the dot's accumulator) put these correct float32
orderings out of the default bound: two-term steps (the 32x32x2 MFMA instruction) 10.3% of elements, FMA'd
product pairs added to the accumulator 10.3%, a k-sequential FMA loop 23.6%, a multiply-then-add loop 23.6%,
2-way split-K 2.5%, 4-way split-K 2 elements; PyTorch's CPU float32 matmul and float64 accumulation pass.
`k = 4` still rejects the FMA loop (2.9%), the two-term orders (0.25%) and 2-way split-K. With `k = 8`, the
worst of them, the FMA and multiply-then-add loops (twice the baseline's error RMS, since they round after
every product), had 19 and 21 of 786,432 elements out of bound (2.4e-5 and 2.7e-5; 3.1e-5 in a 2.1
M-element run with [0, 1) inputs), every other ordering none. The allowance is 2e-4 rather than 1e-4 so the
smallest case absorbs them: `m16` (65,536 elements) then tolerates 13 such elements against about 2
expected, where 1e-4 (6 elements) would fail a correct FMA kernel in about one draw in 200. Modelling the
baseline as the 32x32x2 order or as an FMA loop instead, every correct ordering passed `k = 8` with no
violation. The same bound still rejects every precision downgrade and control tried: TF32-rounded inputs
(43% of elements out of bound), bfloat16 inputs (92%), dropping one k term (100%), a float16 output
rounding (99%) and the fp8 output rounding (100%). Relative output shifts of 4 u and 16 u (the QA auditor's
`shifted_16_ulp` control) stay inside the elementwise bound at this depth but fail the bias test (mean
error 0.49 and 0.92 of its RMS, against an allowance of 5/sqrt(n)), whichever of the three baseline models
is used. Unwritten (NaN) elements are never excused, and the boundary cases keep tail errors visible: one
wrong row of `m4099` is 2.4e-4 of its elements, and the partial last row tiles of `m100`, `m4099` and
`m8205` (4, 3 and 13 rows) are 4%, 7.3e-4 and 1.6e-3.

**Baseline.** No production Triton kernel to copy exists for this operator: eager `torch.matmul` in float32
dispatches to a vendor BLAS library (hipBLASLt or rocBLAS), which a task cannot vendor. The baseline is
therefore a standalone Triton rewrite of the kernel that PyTorch Inductor generates for `torch.mm` under
max-autotune (pytorch/pytorch v2.15.0, commit `d0f2af8b`): the `triton_mm.py.jinja` template (the one Inductor
uses on ROCm with Triton 3.3 or newer) rendered for this layout with a dynamic m (a runtime argument; n, k and
the strides are compile-time constants), Inductor's generated load indices and store suffix, the `mm_grid`
launch grid, `allow_tf32=False` written as `input_precision="ieee"`. Inductor autotunes over its ROCm config
list; without a GPU to autotune on, the baseline fixes one entry of that list per range of m and scales it to
the problem size as Inductor does (block sizes clamped to the next power of two of the dimension, at least 16;
warps capped at one per 16 x 16 output block; an entry's `waves_per_eu` becoming 8 / warps): m <= 64 uses 64 x
16 x 128 tiles with 4 warps and `waves_per_eu=2` (so the 4096 columns give 256 programs; 16 x 16 tiles with 1
warp and `waves_per_eu=8` at m = 16), m <= 256 uses 64 x 32 x 64, m <= 1024 uses 128 x 64 x 64, and larger m
uses 128 x 128 x 32 with 8 warps, all with 2 stages and `matrix_instr_nonkdim=16`. It indexes in int32, as
Inductor does when every operand has fewer than 2^31 elements (always true up to the declared maximum m of
16384). `baseline/LICENSE` is the PyTorch BSD-3-Clause license.

**GPU risks (not yet run).** The per-range configs were not autotuned on MI355X, so the baseline may be well
below the float32 matrix peak, most likely at the range boundaries. The 4 u and 16 u output-shift controls
are rejected by the bias test, not elementwise; that relies on the MI355X float32 MFMA accumulation
rounding to nearest (unbiased), as modelled. If it truncated instead, the baseline's own mean error would
let a shifted result through the bias test's fallback. The violation rates above assume one rounding per
4-term MFMA step; a baseline more accurate than that would tighten the bound on the FMA loop.
