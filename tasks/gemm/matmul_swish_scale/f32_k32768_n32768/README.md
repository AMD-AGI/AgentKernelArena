# Linear + swish + scale, float32, K=N=32768

## Operator

For float32 `x[batch, 32768]`: `out = swish(z) * scaling_factor` with `z = x @ weight.T + bias`
(`nn.Linear(32768, 32768)`), `swish(z) = z * sigmoid(z)` and the scalar `scaling_factor` = 2.0. The GEMM
has float32 accuracy: an IEEE float32 GEMM; TF32 or bfloat16 inputs are a different operator (PyTorch
keeps `allow_tf32` off by default). The output is float32 `[batch, 32768]`.

## Origin and scenario

KernelBench level 2 problem 59 (`Matmul_Swish_Scaling`), the former
`torch2hip/kernelbench/level2/l2n59_Matmul_Swish_Scaling` task. Its module fixes
`in_features = out_features = 32768` and `scaling_factor = 2.0`, and only the batch varies, so the
scenario keeps those constants, float32 and row-major layouts; `batch` is the only var axis.

## Inputs

`x` is uniform on [0, 1), KernelBench's `torch.rand`. `weight` and `bias` are uniform on
`[-1/sqrt(32768), 1/sqrt(32768)]`, the range of PyTorch's default `nn.Linear` initialization for this
fan-in. `scaling_factor` is a scalar input with the module's value 2.0.

## Workloads

KernelBench's `get_inputs` sweep is batch 32 to 512 (16x). The performance cases extend it down to a
single row so that they span 512x: batch 1, 4, 16, 32, 64, 128, 256 and 512. Every batch reads the whole
4 GiB float32 weight (about 1 ms at 4.5 TB/s), so even one row is far above the 10 us floor; from about
64 rows on, the float32 MFMA rate limits (1.1 TFLOP at batch 512). Batches 3, 100 and 1000 are
correctness-only: an odd batch, a batch that is not a multiple of 16 and partial last row tiles (the
minimum, 1, is a performance case).

## Reference

float64 `addmm`, sigmoid and scaling, computed in 4096-feature slices of the weight so that the float64
copy of the weight never exceeds 1 GiB.

## Comparison: `baseline_relative` with `k: 8`, `max_violation_fraction: 0.01` (an override)

The output's error is the float32 rounding of a 32768-term accumulation, and its size depends on the
summation order by up to about 8x. A CPU simulation against the float64 reference modelled the baseline
as the sequential MFMA 16x16x4 chain (four-term dots added to one float32 accumulator in k order, which
is what `acc = tl.dot(a, b, acc)` does; 115 dB SQNR) and compared correct float32 orderings:

- a k-sequential scalar FMA loop (a plain HIP SGEMM; 109 dB) puts 14% of the elements outside the default
  `k = 2` bound and 0.3 to 0.35% outside `k = 8`;
- two-term MFMA steps put 0.09% of the elements outside `k = 8`;
- sixteen-term steps, 4- to 32-way split-K, blocked partial sums and PyTorch's CPU float32 `mm` (all
  more accurate than the baseline) pass `k = 8` with no violations.

`k = 8` with 1% of elements allowed outside the bound accepts the scalar FMA loop with a 3x margin. The
same bound rejects TF32 inputs (95% of the elements out of bound), bfloat16 inputs (99%), the fp8 output
rounding (100%), dropping one k term (97 to 100%) and a scaling factor of 1.999 (98%). The
`shifted_16_ulp` control stays inside the elementwise bound but fails the bias test: the output has a
positive mean, so a relative shift is a biased error. In the simulation it was rejected at batch 4
(z = 0.076 against a threshold of 0.028 on an 8192-column slice) and in 3 of 4 draws at batch 8; the
test's threshold shrinks with the batch while the shift's mean error does not, so the larger cases
reject it reliably. Unwritten (NaN) elements always fail.

Known limitation: bf16x3 split products (three bfloat16 MFMA passes) have an error about 2.5x the
baseline's here (107 dB), close to the scalar FMA loop's 2x, and pass the bound (0.36 to 0.47% of the
elements outside `k = 8`).
For a 32768-term dot the accumulation error dominates, so the bound cannot separate bf16x3 from correct
float32 orderings without rejecting plain FMA loops. (With K = 1024, as in
`tasks/gemm/matmul_swish_bias_groupnorm`, it is rejected.)

Known limitation: the 1% allowance has no cap on the size of the violations, so an error confined to
less than 1% of a case's elements is not rejected by the elementwise test (for example a single wrong
output column with errors of random sign; a consistent-sign error there fails the bias test of the
larger batches once it exceeds about 1e-4). Whole-row errors are rejected: one row is 100%, 33% and 25% of
the elements at batch 1, 3 and 4.

The bound is relative to the baseline's own error, so the baseline is deliberately the least accurate
common float32 order: the Inductor template's `USE_FAST_ACCUM` form `acc = tl.dot(a, b, acc)` pins one
k-sequential accumulation chain. With a blocked baseline (partial dots of 32 to 256 terms added to the
accumulator; 124 to 131 dB in the simulation) the scalar FMA loop would fail `k = 8` on 6 to 29% of the
elements.

## Baseline

No production Triton kernel exists for this module (eager PyTorch runs a vendor BLAS GEMM, which a task
cannot vendor). The baseline is a standalone Triton rewrite of the kernel PyTorch Inductor generates for
it under `torch.compile(mode="max-autotune")` (pytorch/pytorch v2.15.0, commit `d0f2af8b`), without
`torch._inductor` imports: the `triton_mm.py.jinja` template rendered for `addmm(bias, x, weight.t())`
(N, K and the strides are compile-time constants, the batch is a runtime argument), with the addmm
epilogue and the fused `t * sigmoid(t) * scaling_factor` store suffix, the `mm_grid` launch grid,
`allow_tf32=False` written as `input_precision="ieee"`, and the template's `USE_FAST_ACCUM` accumulation
form (above). Inductor renders `addmm` with `USE_FAST_ACCUM=False` (`acc += tl.dot(a, b)`); the baseline
takes the template's other branch on purpose, so that its accumulation order does not depend on how the
compiler treats the separate add.

Inductor autotunes over its ROCm config list; without a GPU to autotune on, the baseline fixes one entry
of `ROCmConfigHeuristic.mm_configs` per batch bucket (batch <= 32: 32 x 32 x 128 tiles, 4 warps,
GROUP_M 8; <= 64: 64 x 32 x 128, 4 warps, GROUP_M 8; <= 128: 64 x 64 x 128, 8 warps, GROUP_M 16; larger:
128 x 128 x 32, 8 warps, GROUP_M 16; all with Inductor's ROCm default of 2 stages and
`matrix_instr_nonkdim=16`). Small batches therefore compute up to 32 rows of which only one is real; they
are bound by the weight read anyway. It indexes in int32, as Inductor does when every operand has fewer
than 2^31 elements (the weight has 2^30). `baseline/LICENSE` is the PyTorch BSD-3-Clause license.

## GPU risks (not yet run)

- The kernel was written and reviewed without a GPU or a Triton install; compilation, the batch-bucket
  configs and timing are unverified. The configs were not autotuned on MI355X.
- The bound was calibrated with a CPU emulation of the MFMA accumulation order. If the GPU baseline is
  much more accurate than the modelled sequential chain, correct FMA-loop candidates may fail; if the
  audit shows that, retune from its data.
- Only the bias test rejects the `shifted_16_ulp` control (its mean error, 16 u times the output's
  mean of about 0.05, is about 4.8e-8). If the GPU baseline's own mean error had the same sign and more
  than about a seventh of that size (the simulated chain's was 2e-9), the `k = 8` fallback of the bias
  test would accept the control; the audit's control suite shows whether that happens.
- Each case holds two 4 GiB weight draws plus a working copy (about 13 GiB) and float64 reference
  temporaries on the device.
