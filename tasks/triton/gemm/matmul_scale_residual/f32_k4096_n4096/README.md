# Linear, scale, residual add, float32, K = N = 4096

**Operator.** `y = x @ weight.T + bias`, `out = y * scaling_factor + y` for `x` [batch, 4096] and an
`nn.Linear` weight [4096, 4096] (`[out_features, in_features]`, row-major) with bias [4096], everything
float32. `scaling_factor` is a float32 scalar input (0.5 in the module). Products and accumulation are IEEE
float32: TF32, bf16x3 or other reduced-precision inputs are a different operator (PyTorch keeps
`allow_tf32` off by default).

**Origin.** KernelBench level 2 problem 40 (`Matmul_Scaling_ResidualAdd`), the former
`torch2hip/kernelbench/level2/l2n40_Matmul_Scaling_ResidualAdd` task, whose baseline was the eager PyTorch
module and whose candidate was HIP. The module fixes `in_features = out_features = 4096` and
`scaling_factor = 0.5` (`get_init_inputs()`); its residual is the linear output itself
(`original_x = x.clone().detach()`), so in inference `out = (1 + scaling_factor) * y`. Its `get_inputs()`
varies the batch over 2048, 4096, 8192, 16384 and 32768. The scenario keeps the constants and the row-major
float32 layout and varies only the batch.

**Inputs.** KernelBench's distributions: `x` is `torch.rand`, uniform on [0, 1); `weight` and `bias` are
uniform on [-1/64, 1/64], the range of PyTorch's default `nn.Linear` initialization for fan-in 4096. `y` then
has a standard deviation of about 1/3.

**Workloads.** Eight performance batches from 256 to 32768 (128x span): the five KernelBench batches plus
256, 512 and 1024, which the original 16x span lacked. Batch 256 is 8.6 GFLOP of float32 matrix work, at
least 55 us at the MI355X float32 matrix peak (about 157 TFLOP/s); batch 32768 is 1.1 TFLOP, about 7 ms at
peak. Batches 1 (the axis minimum), 3, 100 and 1000 are correctness-only boundary shapes (100 and 1000 are
not multiples of 16 or of the 128-row tile). Batch 100 is checked a second time with
`scaling_factor = 2.0` (`b100_s2`), so a kernel that hard-codes the module's 0.5 fails. The largest case
holds 0.5 GB of activation, 0.5 GB of output and about 3 GB of float64 reference temporaries.

**Reference.** `addmm` in float64, then `y * scaling_factor + y` in float64.

**Comparison: `baseline_relative` with `k: 8`, `max_violation_fraction: 5.0e-3` (an override).** The output
is float32 and its error is the rounding of a 4096-deep float32 accumulation, about 10 u (u = 2^-24) of the
output's RMS and some 20x the RMS of one output rounding, not one output rounding. Any other legitimate summation order has an independent error of the same
size, and since that error grows with the magnitude of the running partial sums it is far from uniform
across elements: where the baseline's own error happens to be small, the per-row RMS floor of the default
bound is several times below a correct candidate's error. A CPU emulation of float32 round-to-nearest
accumulation against the float64 reference (32 and 64 rows of the task's distributions, two draws, the
evaluator's own comparison code; the baseline modelled as the 16x16x4 float32 MFMA order, four exact
products added to one float32 accumulator per step in k order, 124.8 dB SQNR) gave, for correct float32
orderings, the fraction of elements outside the bound:

| Candidate ordering | default (`k = 2`) | `k = 8` |
| --- | --- | --- |
| 8-term steps | 0.8% | 0 |
| 2-term steps (the 32x32x2 MFMA instruction) | 8.0% | 0.023% |
| k-sequential FMA loop (a classic HIP SGEMM thread) | 17% | 0.18% |
| 2-way split-K | 0.42% | 0 |
| 4- and 8-way split-K, 32- and 128-deep partial dots, per-block dots added to the accumulator, PyTorch's CPU float32 matmul | <= 0.05% | 0 |

The scalar FMA loop, the most common hand-written HIP accumulation order, therefore needs an allowance
above its 0.18%; 0.5% leaves a 2.7x margin. Every candidate's error mean stayed within the bias test (z below
half its 5-sigma threshold). If the hardware instead rounds after every product (the FMA-loop order), the
baseline is the least accurate of these orderings and every one of them passes with no violation.

The bound still rejects every precision downgrade and control tried (elementwise fraction outside the
`k = 8` bound): TF32-rounded inputs (98%), bfloat16 inputs (99.8%), bf16x3 split products (20%), a float16
output (94%), the control suite's fp8 output (~100%), one dropped k term (99.9%) and a missing bias (~100%).
The control suite's 16-ulp output shift stays inside the elementwise bound and is rejected by the bias test:
its error carries `16 u * mean(out)`, where `mean(out)` (a few thousandths, set by the column sums of the
weight draw) gives a z-score of 0.008 to 0.02 against a threshold of `5 / sqrt(n)` that is 0.0049 at batch 256
and 0.0004 at batch 32768, while a correct float32 ordering is unbiased. Unwritten (NaN) elements are never
excused. The allowance means a handful of wrong elements can pass on the largest batches; the boundary cases
keep row-indexing errors visible (one wrong row is 33% of `b3` and 1% of `b100`).

A whole-tensor `sqnr` bound cannot replace this here: the FMA loop (118.8 dB) and the 16-ulp shift
(119.1 dB) have the same SQNR, so no threshold admits the one and rejects the other.

**Baseline.** No production Triton kernel exists for this module, and eager float32 `nn.Linear` dispatches
to a vendor BLAS library, which a task cannot vendor. The baseline is therefore a standalone Triton rewrite,
written by hand, of the kernel PyTorch Inductor generates for the module under
`torch.compile(mode="max-autotune")` with the GEMM backends restricted to Triton (pytorch/pytorch v2.15.0,
commit `d0f2af8b`; with ATen allowed, autotuning may pick the vendor BLAS instead). Because the addmm feeds
only pointwise users, Inductor's post-grad pass (`should_prefer_unfused_addmm`) turns it into mm + bias add
and fuses the bias add, the scaling and the residual add into the store of its `triton_mm.py.jinja`
template. The template is rendered for `x` and `weight.T` (N, K and the strides are compile-time constants,
the batch is a runtime size, as in Inductor's dynamic-batch kernel), the epilogue is Inductor's store
suffix with the scaling factor as a runtime float32 scalar, and the launch grid is `mm_grid`. Inductor
autotunes over its ROCm `mm_configs`; without a GPU to autotune on, the baseline takes, per batch, the
largest of three entries of that list, `(128, 128, 32)` with 8 warps and GROUP_M 16, `(64, 64, 64)` with 4
warps and GROUP_M 4, `(32, 32, 128)` with 4 warps and GROUP_M 8, that still launches at least 256 tiles
(one per MI355X compute unit), with BLOCK_M clamped to the next power of two of the batch (at least 16) and
the warp count capped as Inductor's `_scale_mm_configs`/`_finalize_mm_configs` do; 2 stages and
`matrix_instr_nonkdim=16` as in Inductor's ROCm defaults; `allow_tf32=False` written as
`input_precision="ieee"`. It indexes in int32, as Inductor does when every operand has fewer than 2^31
elements (always true up to the declared batch maximum). `baseline/LICENSE` is the PyTorch BSD-3-Clause
license.

**GPU risks (not yet run).** The config choice was not autotuned on MI355X, so the baseline may be well
below the float32 matrix peak. The emulation assumes Triton folds the template's `acc += tl.dot(a, b)` into
one accumulator chain (its dot-add combine), as it does for float32; if each BLOCK_K block were summed
separately and then added, the baseline would be about 3x more accurate and single-chain candidates would
exceed the 0.5% allowance. The 16-ulp shift control is rejected through the bias test only, which relies on
the MI355X float32 accumulation rounding to nearest (unbiased), as modelled.
