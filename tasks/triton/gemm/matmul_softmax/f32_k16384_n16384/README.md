# Linear + softmax, float32, K = N = 16384

**Operator.** `out = softmax(x @ weight.T + bias, dim=1)` for `x` [batch, 16384] and an `nn.Linear` weight
[16384, 16384] (`[out_features, in_features]`, row-major) with bias [16384], everything float32, output
float32; the softmax runs over the 16384 output features of each row. Products and accumulation are IEEE
float32: TF32, bf16x3 or other reduced-precision inputs are a different operator (PyTorch keeps
`allow_tf32` off by default).

**Origin.** KernelBench level 2 problem 66 (`Matmul_Dropout_Softmax`), the former
`torch2hip/kernelbench/level2/l2n66_Matmul_Dropout_Softmax` task, whose baseline was the eager PyTorch module
and whose candidate was HIP. The module is built with `get_init_inputs() = [16384, 16384, 0.2]` and was
evaluated in inference (`eval()`) mode, where `nn.Dropout(0.2)` is the identity; the old task's functional
form already omitted it. Its `get_inputs()` varied the batch over 64, 128, 256, 512 and 1024. The scenario
keeps the constants and the row-major float32 layout and varies only the batch.

**Inputs.** KernelBench's distributions: `x` is `torch.rand`, uniform on [0, 1); `weight` and `bias` are
uniform on [-1/128, 1/128], the range of PyTorch's default `nn.Linear` initialization for fan-in 16384. The
logits then have a standard deviation of about 1/3, so each softmax row is fairly flat.

**Workloads.** Eight performance batches from 8 to 1024 (128x span): the five KernelBench batches plus 8, 16
and 32, which the original 16x span lacked. Every call streams the 1 GiB weight, at least about 240 us at
the MI355X's achievable HBM bandwidth even for batch 8; batch 1024 is 0.55 TFLOP of float32 matrix work,
about 3.5 ms at the float32 matrix peak. Batches 1 (the axis minimum), 3, 100 and 1000 are correctness-only
boundary shapes (100 and 1000 are not multiples of 16 or of the baseline's row tiles). The float64 reference
needs a 2 GiB float64 copy of the weight.

**Reference.** `addmm` in float64, then `torch.softmax` over the feature axis in float64.

**Comparison: `sqnr` with `min_db: 120.5` (an override).** The output error of a correct float32
implementation is dominated by the rounding of the 16384-deep float32 accumulation of the logits (about 6 u
of absolute logit error for the baseline's order, u = 2^-24, which becomes the same relative error of each
probability), not by one output rounding, and the summation order of each row's normalizer adds one common
relative error to the whole row. The default bound fails on both counts. Elementwise, a different correct
accumulation order puts 5% (2-term MFMA steps) to 11% (a scalar FMA loop) of the elements outside `k = 2`.
Its bias test reads the row-common normalizer error as a systematic bias and then compares it with the
baseline's own, randomly small, normalizer error: in a Monte Carlo of 100 draws each at 1, 8 and 64 rows,
correct candidates (a different tree or tile-atomic normalizer, or only a different GEMM order) failed
`baseline_relative` even with `k = 8` and a 0.5% allowance in 4% to 23% of calls, so a correct candidate
would almost surely fail one of the 24 checked calls.

SQNR is the whole-tensor rule that works. A CPU emulation of float32 round-to-nearest arithmetic against the
float64 reference (1, 4 and 16 rows of the task's distributions, several draws, the evaluator's own SQNR
code; the baseline modelled as 4 exact products added to one float32 accumulator per step in k order, then
Inductor's 2048-lane online softmax) gave:

| Implementation | SQNR (dB) |
| --- | --- |
| baseline model (4-term MFMA steps, Inductor online softmax) | 126.9 to 127.4 |
| 4-term steps with a two-pass tree, exp2-based exp and reciprocal multiply, per-block online merge, or 64 or 256 atomically added tile partials | 125.8 to 127.2 |
| 2-term steps (the 32x32x2 MFMA instruction), any of those normalizers | 123.6 to 124.2 |
| scalar FMA loop (a classic HIP SGEMM thread), any of those normalizers | 120.8 to 121.5 |
| 2-way and 4-way split-K, per-block dots added to the accumulator, 128-deep partial dots, PyTorch's CPU float32 matmul | 128.4 to 138.3 |
| single-thread sequential normalizer (row-sum error about 30 u), any GEMM order | 114.0 to 116.3 (rejected) |

The threshold sits between the framework's 16-ulp output-shift control and the least accurate correct
ordering: the shift multiplies every element by `1 + 2^-20`, so its SQNR cannot exceed 120.41 dB (measured
117.4 to 120.2 dB), while the scalar FMA loop with 256 atomically added normalizer partials, the worst
correct combination emulated, never fell below 120.84 dB on a single row. `min_db: 120.5` allows an RMS
error of about 15.8 u. It also rejects every precision downgrade tried: bf16x3 split products (117.5 dB),
TF32-rounded inputs (81 dB), a float16 output (71 dB), bfloat16 inputs (63 dB), one dropped k term (52 dB),
a missing bias (47 dB) and fp8 output rounding; an 8-ulp shift of the modelled baseline's output (123.7 to
124.0 dB) is within the error of correct float32 orderings and passes. Unwritten (NaN) elements are never excused. SQNR is a
whole-tensor measure but a strict one at this level: one element wrong by its own magnitude costs 42 dB in
`b1` and 72 dB in `b1024`, and one wrong row in `b100` is 20 dB.

**Baseline.** No production Triton kernel exists for this module, and eager float32 `nn.Linear` dispatches
to a vendor BLAS library, which a task cannot vendor. The baseline is therefore a standalone Triton rewrite,
written by hand, of the two kernels PyTorch Inductor generates for the module under
`torch.compile(mode="max-autotune")` with the GEMM backends restricted to Triton (pytorch/pytorch v2.15.0,
commit `d0f2af8b`), since no GPU was available to run Inductor:

1. The GEMM. The addmm feeds the softmax's max reduction, which is not a pointwise use, so the post-grad
   pass keeps the addmm (`should_prefer_unfused_addmm`) and `tuned_addmm` lowers it to the
   `triton_mm.py.jinja` template with `addmm_epilogue` (`acc + bias`) as its store suffix, writing the logits
   to an intermediate buffer that `run()` allocates on each call, as Inductor's wrapper does. The template
   is rendered for `x` and `weight.T` (N, K and the strides are compile-time constants, the batch a runtime
   size), the grid is `mm_grid`, and one entry of Inductor's ROCm `mm_configs` is chosen per batch instead
   of autotuning: the largest of `(128, 128, 32)` with 8 warps, `(64, 64, 64)` with 4 warps and
   `(32, 32, 128)` with 4 warps that still launches at least 256 tiles, scaled to the batch as
   `_scale_mm_configs`/`_finalize_mm_configs` do; 2 stages, `matrix_instr_nonkdim=16`; `allow_tf32=False`
   written as `input_precision="ieee"`.
2. The softmax. With `config.online_softmax` Inductor generates one looped reduction kernel per row: a first
   loop folds 2048-wide blocks into per-lane running (max, sum) pairs with
   `triton_helpers.online_softmax_combine`, `online_softmax_reduce` combines the lanes, and a second loop
   writes `exp(y - max) / sum` (ROCm keeps per-lane vector accumulators). The helpers are copied in with
   their default `use_fast_math=False` and `strict_signed_zero=False`, so the exponentials are
   `libdevice.exp`. Launch: one row per program, 2048-wide blocks, 8 warps.

`baseline/LICENSE` is the PyTorch BSD-3-Clause license.

**GPU risks (not yet run).** The block configs were not autotuned on MI355X, so the baseline may be well
below the float32 matrix peak, and small batches launch only 256 GEMM tiles over the 1 GiB weight. The
threshold leaves 0.3 to 1.0 dB to scalar-FMA-loop candidates and is calibrated on emulated rounding: if the
hardware's float32 MFMA rounds after every product rather than once per 4-product step, the baseline itself
scores about 121 dB, still above the threshold; a baseline below 120.5 dB on MI355X would fail its own
correctness check, and the threshold would need recalibrating there.
