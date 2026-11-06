# GEMM, float32, tall-skinny, k=16 n=16384

Operator: `out = a @ b` (`torch.matmul`) for a tall `a` [m, 16] and a wide `b` [16, 16384], both row-major
float32, output float32 [m, 16384]. The contraction is only 16 deep, so the cost is writing the output (64
KB per row), not the arithmetic. Products and accumulation are IEEE float32: TF32 or other reduced-precision
inputs are a different operator.

**Origin.** KernelBench level 1 problem 9 (tall-skinny matrix multiplication), the former
`torch2hip/kernelbench/level1/l1n9_Tall_skinny_matrix_multiplication_` task: `A` [M, K] @ `B` [K, M] with
(M, K) = (8192, 16), (16384, 16), (32768, 32), (65536, 16) and (32768, 64), a square M x M output. A
scenario fixes everything but one axis, so this task takes the problem's dominant inner size k = 16 and the
output width of its (16384, 16) case, n = 16384, and varies m, the tall side, which the problem escalated up
to 65536.

**Workloads.** Eight timed row counts from 640 to 65536 (102x span; 42 MB to 4.3 GB of output writes). The
square outputs of the original cases cannot give the required 100x span: with n = m, a 100x span in m is a
10,000x span in output size, from about 27 MB (the 10 us floor) to 270 GB, and the original m = 65536 case
alone writes 17 GB, more than the evaluator can hold with its float64 references and comparison temporaries.
With n fixed, the KernelBench tall sides 8192, 16384, 32768 and 65536 are all timed, the 16384 case being
the original case_1 exactly (the 32768 cases had k = 32 and 64 there). The smallest timed case is sized for
the floor: 42 MB of writes, about 13 us by the bytes / 4.5 TB/s + 4 us estimate. Correctness-only boundary
shapes: 1 (the axis minimum), 3 (not a power of two), 100 (not a multiple of 16) and 5000 (a partial last
row tile). Inputs keep KernelBench's `torch.rand` distribution, uniform on [0, 1).

**Reference.** `torch.matmul` in float64.

**Comparison: `baseline_relative` with `k: 8`, `max_violation_fraction: 1.0e-4` (an override).** The output is
float32 and each element is a 16-term float32 dot, so the baseline's error (an RMS relative error of 0.6 u, u
= 2^-24 being the float32 unit roundoff; 149 dB SQNR in simulation) is the same order as one output rounding,
and a different legitimate summation order differs from it by about as much. A CPU simulation against the
float64 reference (uniform [0, 1) inputs, 2,097,152 output elements; the baseline modelled as the MFMA 16x16x4
order the Triton dot uses for a single 16-deep step, four-term dots added to a float32 accumulator) put these
correct float32 orderings out of the default bound (`k = 2`, no violations): two-term steps (the 32x32x2 MFMA
instruction) 0.22% of elements, a k-sequential FMA loop 2.4%, a multiply-then-add loop 2.7%, PyTorch's CPU
float32 matmul 2.4%, pairwise summation 0.09%, 2-way split-K 0.002%; 8- and 16-term steps and float64
accumulation pass. `k = 4` still rejects the FMA and multiply-then-add loops (0.12% and 0.15%). With `k = 8`
only the multiply-then-add loop had any element out of bound (one, 4.8e-7 of them), and modelling the baseline
as the 32x32x2 order or as an FMA loop instead, every correct ordering passed with none; the allowance of 1e-4
lets the one-row `m1` case (16,384 elements) absorb a single such element. The same bound rejects every
precision downgrade and control tried: the 16-ulp output shift (99.9% of elements out of bound), the 4-ulp
shift (0.8%, and the bias test), TF32-rounded inputs (99.6%), bfloat16 inputs (99.96%), dropping one k term
(100%) and the fp8 output rounding (100%). Unwritten (NaN) elements are never excused, and a wrong partial
tile in the boundary cases is far above the allowance (one wrong row of `m5000` is 2e-4 of its elements). The
bias test is strictest on the largest case: `m65536` has 1.07e9 elements, so a mean error above 5/sqrt(n) =
1.5e-4 of the error RMS fails unless the baseline's own mean error covers it. In a full-size simulation
(65,536 x 16,384, the baseline modelled as above) the FMA loop's mean error was -5.6e-5 of its RMS and 2-way
split-K's -3.6e-5, both passing, while the 16 u shift was 0.98. KernelBench's uniform [0, 1) inputs are kept:
with a 16-deep contraction the accumulator stays below 16 and its ulp at most 2^-20, so products small enough
to bias a long FMA loop toward rounding down (the reason the n = k = 4096 scenario draws [0.5, 1.5) instead)
are too rare to matter here.

**Baseline.** No production Triton kernel to copy exists for this operator: eager `torch.matmul` in float32
dispatches to a vendor BLAS library (hipBLASLt or rocBLAS), which a task cannot vendor. The baseline is
therefore a standalone Triton rewrite of the kernel that PyTorch Inductor generates for `torch.mm` under
max-autotune (pytorch/pytorch v2.15.0, commit `d0f2af8b`): the `triton_mm.py.jinja` template (the one Inductor
uses on ROCm with Triton 3.3 or newer) rendered for this layout with a dynamic m (a runtime argument; n, k and
the strides are compile-time constants), Inductor's generated load indices and store suffix, the `mm_grid`
launch grid, `allow_tf32=False` written as `input_precision="ieee"`. Inductor autotunes over its ROCm config
list; without a GPU to autotune on, the baseline fixes one entry of that list (128 x 128 x 32 tiles, 8 warps,
2 stages, GROUP_M 16, `matrix_instr_nonkdim=16`) and scales it to the problem size as Inductor does, which
turns BLOCK_K into 16 = k: one k step per tile (and 16-row tiles for the m < 16 boundary cases). It indexes in
int32, as Inductor does when every operand has fewer than 2^31 elements (the largest output, 65536 x 16384,
has 2^30). `baseline/LICENSE` is the PyTorch BSD-3-Clause license.

**GPU risks (not yet run).** The timing floor of the smallest case: `m640` writes 42 MB, estimated at about
13 us, but if the MI355X memory-side cache absorbs those writes faster than HBM bandwidth it can fall below
10 us, and a larger smallest case would break the 100x span (the largest case is already at the memory the
comparison can hold). The 4.3 GB `m65536` output needs about 150 GB of device memory for its two float64
references, the baseline copies and the comparison temporaries. The fixed config was not autotuned on
MI355X, and for this store-bound product the efficiency of its output stores from the MFMA
accumulator layout decides its speed.
