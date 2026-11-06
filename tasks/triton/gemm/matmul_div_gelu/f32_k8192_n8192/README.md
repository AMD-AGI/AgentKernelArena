# Linear, divide, GELU, float32, K = N = 8192

**Operator.** `out = gelu((x @ weight.T + bias) / divisor)` for `x` [batch, 8192] and an `nn.Linear` weight
[8192, 8192] (`[out_features, in_features]`, row-major) with bias [8192], everything float32. The GELU is the
exact (erf) form, `0.5 * y * (1 + erf(y / sqrt(2)))`, as `F.gelu` with `approximate="none"`. `divisor` is a
float32 scalar input (10 in the module). Products and accumulation are IEEE float32: TF32, bf16x3 or other
reduced-precision inputs are a different operator (PyTorch keeps `allow_tf32` off by default).

**Origin.** KernelBench level 2 problem 86 (`Matmul_Divide_GELU`), the former
`torch2hip/kernelbench/level2/l2n86_Matmul_Divide_GELU` task, whose baseline was the eager PyTorch module and
whose candidate was HIP. The module fixes `input_size = output_size = 8192` and `divisor = 10.0`; its
`get_inputs()` varies the batch over 256, 512, 1024, 2048 and 4096. The scenario keeps the constants and the
row-major float32 layout and varies only the batch.

**Inputs.** KernelBench's distributions: `x` is `torch.rand`, uniform on [0, 1); `weight` and `bias` are uniform
on [-1/sqrt(8192), 1/sqrt(8192)], the range of PyTorch's default `nn.Linear` initialization for this fan-in.
The linear output then has a standard deviation of about 1/3 and the GELU input (after the division by 10)
about 0.033.

**Workloads.** Nine performance batches from 16 to 4096 (256x span): the five KernelBench batches plus 16, 32,
64 and 128. Every case reads the 268 MB weight, so even batch 16 is estimated at well over 10 us on MI355X
(about 60 us of weight traffic at 4.5 TB/s); batch 4096 is 550 GFLOP of float32 matrix work, several
milliseconds. Batches 1 (the axis minimum), 3, 100 and 1000 are correctness-only boundary shapes; one more
correctness case (`b100_div4`) uses `divisor = 4` so the scalar input cannot be hard-coded.

**Reference.** `addmm` in float64, the division, and the erf GELU in float64.

**Comparison: `baseline_relative` with `k: 8`, `max_violation_fraction: 5.0e-3` (an override).** The output
is float32 and its error is the rounding of the 8192-deep float32 accumulation, an RMS about 13 times one
output rounding, not one output rounding. Any other legitimate summation order has an independent error of
the same size, and since that error grows with the running partial sums it is far from uniform across
elements: where the baseline's own error happens to be small, the per-row RMS floor of the bound is several
times below a correct candidate's error. A CPU simulation against the float64 reference (the task's input
distributions at K = 8192; 64 full rows of 8192 outputs and 128 rows of 4096; the baseline modelled as the
16x16x4 float32 MFMA order, four exact products added to the float32 accumulator per step in k order) gave,
for correct float32 orderings, the fraction of elements outside the bound:

| Candidate ordering | default (`k = 2`) | `k = 8` |
| --- | --- | --- |
| k-sequential FMA loop (a classic HIP SGEMM thread), or multiply-then-add | 17% | 0.19% (worst row 0.29%) |
| 2-term steps (the 32x32x2 MFMA instruction, `matrix_instr_nonkdim=32`) | 8.2% | 0.02% |
| 2-way split-K | 0.50% | 0 |
| 16-term steps | 0.16% | 0 |
| 4-way split-K | 0.05% | 0 |
| PyTorch's CPU float32 matmul, 128-deep exact partial dots | <= 0.0002% | 0 |

The FMA loop, the most common hand-written accumulation order, has twice the baseline's error RMS (it rounds
after every product) and needs an allowance above its 0.19%; 0.5% leaves a 2.6x margin overall and 1.7x on
the worst single row (the one-row case `b1`). The same bound still rejects every precision downgrade and
control tried (64 x 8192 run): TF32-rounded inputs (98% of elements out of bound), bfloat16 inputs (99.7%),
bf16x3 split products (10%), one dropped k term (99.8%), a missing bias (99.98%), a float16 output (92%) and
an fp8 output (~100%). The tanh-approximate GELU stays inside the elementwise bound but fails the bias test
(z-score 0.084 against the `5 / sqrt(n)` limit of 0.0069). The control suite's 16-ulp output shift is also
rejected by the bias test only (z 0.013 against 0.0069 at 524,288 elements; the limit falls to 0.00086 at
batch 4096), which relies on correct float32 accumulation being unbiased (round to nearest). Unwritten (NaN)
elements are never excused. The allowance means a few wrong elements can pass on large batches; the
boundary cases keep row-indexing errors visible (one wrong row is 33% of `b3` and 1% of `b100`).

**Baseline.** No production Triton kernel exists for this module, and eager float32 `nn.Linear` dispatches to a
vendor BLAS library, which a task cannot vendor. The baseline is therefore a standalone Triton rewrite of the
kernel PyTorch Inductor generates for the module under `torch.compile(mode="max-autotune")` with the GEMM
backends restricted to Triton (pytorch/pytorch v2.15.0, commit `d0f2af8b`; with ATen allowed, autotuning may
pick the vendor BLAS instead): the `triton_mm.py.jinja` mm template that `tuned_mm` autotunes, rendered for
`x` and `weight.T` (N, K and the strides are compile-time constants, the batch is a runtime size, as in
Inductor's dynamic-batch kernel), with the epilogue Inductor fuses into the template store: the bias add
(the post-grad pass `should_prefer_unfused_addmm` splits an addmm whose users are all pointwise into mm and
add), the division by the constant divisor turned into a multiplication by its reciprocal
(`div_prim`; here `1 / divisor` is computed on the host and passed as a float32 scalar) and the GELU
decomposition `y * 0.5 * (1 + erf(y * M_SQRT1_2))` with `libdevice.erf`. The launch grid is `mm_grid`.
Inductor autotunes over its ROCm `mm_configs`; without a GPU to autotune on, the baseline picks one entry per
batch range instead: `(32, 32, 128)` tiles with 4 warps and GROUP_M 8 up to batch 64, `(64, 64, 64)` with 4 warps
and GROUP_M 4 up to batch 256 and `(128, 128, 32)` with 8 warps and GROUP_M 16 above, so that every timed
batch launches at least 256 tiles; BLOCK_M is clamped to the next power of two of the batch (at least 16) as
Inductor's `_scale_mm_configs` does; 2 stages and `matrix_instr_nonkdim=16` as in Inductor's ROCm defaults;
`allow_tf32=False` written as `input_precision="ieee"`. It indexes in int32, as Inductor does when every operand
has fewer than 2^31 elements (always true up to the declared batch maximum of 16384). `baseline/LICENSE` is
the PyTorch BSD-3-Clause license.

**GPU risks (not yet run).** The per-batch configs were not autotuned on MI355X, so the baseline may be well
below the float32 matrix peak, most likely near the batch-range boundaries; every timed case is still far
above 10 us (batch 16 reads the whole 268 MB weight). The bound was calibrated on a CPU model of the MFMA
accumulation order: if the hardware's float32 MFMA rounds more often than modelled, the baseline's error and
the bound grow (safe for candidates); if Triton does not fold `acc += tl.dot(...)` into the MFMA accumulator
and adds each BLOCK_K partial dot separately (a more accurate order), a k-sequential FMA candidate would
exceed the 0.5% allowance. The QA auditor does not exercise alternative correct orderings, so this would show
only as unexpected candidate failures. The 16-ulp control is rejected through the bias test on the large
batches, as modelled. One kernel launch per call, no host synchronization: CUDA-graph capture should hold.
