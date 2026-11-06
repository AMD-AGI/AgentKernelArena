# Linear, AvgPool1d, GELU, scale, row max, float32, K = N = 8192

**Operator.** For `x` [batch, 8192] and an `nn.Linear` weight [8192, 8192] (`[out_features, in_features]`,
row-major) with bias [8192], everything float32:

1. `y = x @ weight.T + bias` ([batch, 8192]);
2. `p = avg_pool1d(y, 16)`: the mean of each window of 16 consecutive features (stride 16, no padding),
   [batch, 512];
3. `g = gelu(p)`, the exact (erf) form `0.5 * p * (1 + erf(p / sqrt(2)))` (`F.gelu`, `approximate="none"`);
4. `out[b] = max_j(g[b, j] * scale_factor)`, the maximum over the 512 pooled values of the row, [batch].

`scale_factor` is a float32 scalar input (2 in the module). Products and accumulation are IEEE float32: TF32,
bf16x3 or other reduced-precision inputs are a different operator (PyTorch keeps `allow_tf32` off by default).

**Origin.** KernelBench level 2 problem 98 (`Matmul_AvgPool_GELU_Scale_Max`), the former
`torch2hip/kernelbench/level2/l2n98_Matmul_AvgPool_GELU_Scale_Max` task, whose baseline was the eager PyTorch
module and whose candidate was HIP. The module fixes `in_features = out_features = 8192`,
`pool_kernel_size = 16` and `scale_factor = 2.0`; its `get_inputs()` varies the batch over 256, 512, 1024, 2048
and 4096. The scenario keeps the constants and the row-major float32 layout and varies only the batch.

**Inputs.** KernelBench's distributions: `x` is `torch.rand`, uniform on [0, 1); `weight` and `bias` are uniform
on [-1/sqrt(8192), 1/sqrt(8192)], the range of PyTorch's default `nn.Linear` initialization for this fan-in.
The linear output has a standard deviation of about 1/3, the pooled values about 0.083, and each output (the
largest of 512 pooled values after GELU and scaling) lies around 0.2 to 0.4.

**Workloads.** Nine performance batches from 16 to 4096 (256x span): the five KernelBench batches plus 16, 32,
64 and 128. Every case reads the 268 MB weight, so even batch 16 is estimated at well over 10 us on MI355X
(about 60 us of weight traffic at 4.5 TB/s); batch 4096 is 550 GFLOP of float32 matrix work, several
milliseconds, plus a 134 MB linear-output temporary written and read back. Correctness-only boundary shapes:
100 and 1000 (not powers of two, not multiples of 16: partial last row tiles of the medium- and large-tile
configurations), and 1000 with `scale_factor = -1.5`, so the scalar cannot be hard-coded and the maximum must
be taken after the scaling (with a negative scale it selects the smallest GELU value of the row). The
negative-scale case uses batch 1000, where the 0.2% allowance applies, rather than 100, where a correct
FMA-loop kernel would fail it in about 1.7% of calls (see the comparison below). The axis minimum, and the
smallest timed batch, is 16 rather than 1: the operator writes one value per row, so a batch of n gives the
comparison only n elements, and the baseline-relative bound takes its RMS floor from those n baseline errors.
With one row that floor is the baseline's own error of that element, which is small by chance often enough
that a correct candidate with an independent rounding error of the same size falls outside eight times it in
about 6% of calls (12% to 14% for a k-sequential FMA loop); see the comparison below for batch 16.

**Reference.** `addmm` in float64, the 16-wide mean, the erf GELU, the scale and `amax` over the pooled axis,
all in float64.

**Comparison: `baseline_relative` with `k: 8`, `max_violation_fraction: 2.0e-3` (an override).** The output is
float32 and its error is the rounding of the 8192-deep float32 accumulation carried through the pooling, GELU,
scale and maximum, an RMS about 6 times one output rounding, not one output rounding. Any other legitimate
summation order has an independent error of the same order, and the default bound (`k = 2`, no violations)
rejects it. A CPU simulation against the float64 reference (512 rows with the task's input distributions; the
baseline modelled as the 16x16x4 float32 MFMA order, four exact products added to the float32 accumulator per
step in k order, then the baseline's left-to-right pooling) gave correct float32 orderings these error RMS
ratios to the baseline: k-sequential FMA loop 1.9 (it rounds after every product), 2-term steps (the 32x32x2
MFMA instruction) 1.35, 4-way split-K 0.50, PyTorch's CPU float32 matmul 0.36, each also with a pairwise
instead of left-to-right 16-term pool sum. With one output value per row the bound's RMS floor comes from as
many baseline errors as there are rows: at batch 1 every one of these orderings fails `k = 8` in 2% to 14% of
calls (hence the axis minimum of 16). A Gaussian model of the errors underestimates the FMA loop's tail, so a
second simulation measured it directly: ten weight draws, 28,672 rows in all, with only each row's winning
pool window accumulated in float32 (the other windows cannot change the maximum; on a 400-row draw this
reproduces the full simulation exactly). Even with the RMS floor of a whole 2048- or 4096-row draw, 7e-5 of
the FMA loop's rows fall outside the bound (the Gaussian model gives 1.3e-5 beyond eight baseline RMS), and
under this bound the FMA loop fails 0.6% of the 16-row calls, 0.4% at 32 rows, 0.9% at 64, 0.7% at 100 (1.7%
with the negative scale), 1.3% at 128 and 2.7% at 256, and none from 512 rows up, where the 0.2% allowance
excuses at least one row. Over the task's cases and two draws per case that leaves roughly a 10% to 15% chance
that a correct FMA-loop kernel (a SIMT HIP SGEMM) fails one case. The 2-term and 4-way split-K orders failed
no call of 16 rows or more, so kernels built on MFMA instructions (`tl.dot`, or HIP MFMA intrinsics) are not
at risk in the model. `k` cannot exceed 8, and batches below 500 have no allowance to give. A 0.4% allowance
would remove the batch-256 failures but would excuse one wrong row from batch 250 up, so the task keeps 0.2%.
The bound still rejects every precision downgrade and control tried (512-row run): TF32-rounded inputs (98% of
rows out of bound), bfloat16 inputs (99.6%), bf16x3 split products (3%; rejected on every 100-row batch), one
dropped k term (99.6%), a missing bias (100%), 8-wide pooling (100%), the tanh-approximate GELU (97%), a
float16 output (99%), an fp8 output (100%) and, with the negative scale, taking the maximum before scaling
(100%). The control suite's 16-ulp output shift stays inside the elementwise bound and is rejected by the bias
test from batch 100 up (z-score 0.93 against `5 / sqrt(n)`, 0.5 at batch 100 and 0.078 at batch 4096), which
relies on correct float32 accumulation being unbiased. Unwritten (NaN) elements are never excused; the
allowance excuses no element below batch 500 and 8 of the 4096 rows at batch 4096.

**Baseline.** No production Triton kernel exists for this module, and eager float32 `nn.Linear` dispatches to a
vendor BLAS library, which a task cannot vendor. The baseline is therefore a standalone Triton rewrite of the
two kernels PyTorch Inductor generates for the module under `torch.compile(mode="max-autotune")` with the GEMM
backends restricted to Triton (pytorch/pytorch v2.15.0, commit `d0f2af8b`; with ATen allowed, autotuning may
pick the vendor BLAS for the GEMM instead):

- `_triton_tem_fused_addmm`: the `triton_mm.py.jinja` mm template that `tuned_addmm` autotunes, rendered for
  `x` and `weight.T` (N, K and the strides are compile-time constants, the batch is a runtime size, as in
  Inductor's dynamic-batch kernel), with the bias add (`addmm_epilogue`) fused into its store. The pooling
  reads 16 neighbouring GEMM outputs, which a template epilogue cannot, so Inductor materializes the linear
  output; the baseline allocates it as a float32 temporary in `run()`. The launch grid is `mm_grid`.
  Inductor autotunes over its ROCm `mm_configs`; without a GPU to autotune on, the baseline picks one entry per
  batch range instead: `(32, 32, 128)` tiles with 4 warps and GROUP_M 8 up to batch 64, `(64, 64, 64)` with
  4 warps and GROUP_M 4 up to batch 256 and `(128, 128, 32)` with 8 warps and GROUP_M 16 above, so that every
  timed batch launches at least 256 tiles; BLOCK_M is clamped to the next power of two of the batch (at least 16) as
  Inductor's `_scale_mm_configs` does; 2 stages and `matrix_instr_nonkdim=16` as in Inductor's ROCm defaults;
  `allow_tf32=False` written as `input_precision="ieee"`; int32 indexing, as Inductor uses when every operand
  has fewer than 2^31 elements.
- `_triton_per_fused_avg_pool2d_gelu_max_mul`: the persistent reduction Inductor generates for the rest.
  `avg_pool1d` decomposes to `avg_pool2d` with a (1, 16) window, which `_avg_poolnd` unrolls (window <= 25)
  into a left-to-right float32 sum of the 16 loads times 1/16; then the GELU decomposition
  `p * 0.5 * (1 + erf(p * M_SQRT1_2))` with `libdevice.erf`, the multiplication by `scale_factor` and the
  NaN-propagating row maximum (`triton_helpers.max2` / `maximum`, inlined). The block is fixed at 8 rows of
  the 512 pooled values (Inductor's persistent-reduction limit of 4096 elements) and 8 warps.

`baseline/LICENSE` is the PyTorch BSD-3-Clause license.

**GPU risks (not yet run).** The per-batch configs were not autotuned on MI355X, so the baseline GEMM may be
well below the float32 matrix peak; every timed case is still far above 10 us (batch 16 reads the whole
268 MB weight). The bound was calibrated on a CPU model of the MFMA accumulation order: if Triton does not
fold `acc += tl.dot(...)` into the MFMA accumulator and adds each BLOCK_K partial dot separately (a more
accurate order), correct but less accurate candidates (the FMA loop) fail more often than estimated above;
the QA auditor does not exercise alternative correct orderings, so this would show only as unexpected
candidate failures. The 16-ulp control is rejected through the bias test, as modelled. Each call launches
two kernels and allocates the float32 linear-output temporary with `torch.empty`; neither synchronizes, so
CUDA-graph capture should hold.
