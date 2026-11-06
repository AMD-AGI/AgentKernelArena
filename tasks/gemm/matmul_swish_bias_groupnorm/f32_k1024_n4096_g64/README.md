# Linear + swish + bias + GroupNorm, float32, K=1024, N=4096, 64 groups

## Operator

For float32 `x[batch, 1024]`:

1. `z = x @ weight.T + bias` (`nn.Linear(1024, 4096)`), with float32 accuracy: an IEEE float32 GEMM.
   TF32, bfloat16 or bf16x3 inputs are a different operator (PyTorch keeps `allow_tf32` off by default).
2. `h = z * sigmoid(z) + extra_bias` (swish, then a per-feature bias of shape `[4096]`).
3. GroupNorm with 64 groups of 64 contiguous features per row: per row and group, the mean and the
   biased variance of the 64 values, `(h - mean) * rsqrt(var + epsilon)`, then the per-feature affine
   `* gn_weight + gn_bias`. `epsilon` (1e-5, `nn.GroupNorm`'s default) is a scalar input.

The output is float32 `[batch, 4096]`.

## Origin and scenario

KernelBench level 2 problem 37 (`Matmul_Swish_Sum_GroupNorm`), the former
`torch2hip/kernelbench/level2/l2n37_Matmul_Swish_Sum_GroupNorm` task. Its module fixes
`in_features=1024, out_features=4096, num_groups=64, bias_shape=(4096,)` and only the batch varies, so
the scenario keeps those constants, float32 and row-major layouts; `batch` is the only var axis.

## Inputs

- `x` is uniform on [0, 1), KernelBench's `torch.rand`.
- `weight` and `bias` are uniform on `[-1/32, 1/32]`, the range of PyTorch's default `nn.Linear`
  initialization for 1024 input features; `extra_bias` is standard normal, KernelBench's `torch.randn`.
- KernelBench leaves `nn.GroupNorm` at its initialization (`weight = 1`, `bias = 0`). Constant draws are
  degenerate for the QA auditor and would hide a missing or misindexed affine, so the task draws
  `gn_weight ~ U[0.5, 1.5]` and `gn_bias ~ N(0, 0.5)`, as for trained parameters.

## Workloads

KernelBench's `get_inputs` sweep is batch 4096 to 65536 (16x). The performance cases extend it down to
256 so that they span 256x: batch 256, 512, ..., 65536 (nine cases, 2.1 to 563 GFLOP). Even the smallest
needs about 14 us at the MI355X float32 matrix peak (256 tiles of 64 x 64, each a 1024-deep dot), so it
is expected well above the 10 us floor. Batches 1, 3, 100, 128 and 1000 are correctness-only: the
minimum, an odd batch, a batch that is not a multiple of 16, and partial last row tiles.

## Comparison: `sqnr` with `min_db: 130` (an override)

The default `baseline_relative` bound is not usable here; its bias test rejects correct float32
implementations:

- `extra_bias` and `gn_bias` are per-feature constants, identical in every row. When such a constant is
  added in float32 to a larger varying term, its low bits are rounded the same way in every row, so part
  of the float32 rounding error is a fixed per-column offset. The bias test assumes errors that are
  independent across all `batch * 4096` elements; with per-column structure its threshold
  (`5 / sqrt(n)`) shrinks with the batch while the true mean error does not.
- In a CPU simulation against the float64 reference (baseline: a float32 GEMM, the Inductor epilogue
  and two-pass group statistics, all in float32), PyTorch's own CPU float32 `group_norm` applied to the
  same float32 GEMM output failed the default bias test, even with `k = 8`, in 6 of 24 draws at batch 256 and 1 of 4 at
  batch 4096 (its mean error up to 200x the baseline's in a draw where the baseline's happened to be
  near zero). A float32 sum with a per-column constant addend showed the same effect in isolation
  (a 3.6-sigma mean error over 4 million sums where random addends gave 0.5 sigma).
- The elementwise part would also need an override: a k-sequential scalar FMA loop (a plain HIP SGEMM)
  puts 4 to 7% of the elements outside the default `k = 2` bound and up to 0.17% outside `k = 8`.

`sqnr` compares the total error power with the signal and is insensitive to that structure. In a
simulation with the baseline modelled as the MFMA 16x16x4 order (four-term float32 dots added to a
float32 accumulator in k order, then the Inductor epilogue and two-pass group statistics; batches 1, 3,
64 and 256), every correct float32 variant tried reached 137.6 to 143.6 dB: k-sequential scalar FMA,
two-term and sixteen-term MFMA steps, blocked and 2- to 8-way split-K accumulation, PyTorch's CPU
float32 `mm`, Welford and sum-of-squares statistics, PyTorch's CPU float32 `group_norm`, `1 / sqrt`
instead of `rsqrt`, and `1 / (1 + exp(-z))` for the sigmoid (the modelled baseline: 141.1 to
142.0 dB across draws and two emulations of the MFMA step). The 130 dB threshold rejects, with
margin: the `shifted_16_ulp` control (120.4 dB),
bf16x3 split products (124.6 dB), TF32 inputs (88 dB), bfloat16 inputs (70 dB), the fp8 output rounding
(32 dB), `epsilon = 1e-6` (107 dB), the unbiased variance (43 dB) and dropping one k term (47 to
54 dB). A split-bfloat16 emulation that represents the float32 inputs exactly (bf16x6, three pieces and
six products) reached 142 dB and passes: it is float32-accurate. Unwritten (NaN) and non-finite
elements always fail, and a single element off by 1 already drops the SQNR of the largest case to
85 dB.

## Baseline

No production Triton kernel exists for this module (eager PyTorch runs a vendor BLAS GEMM, which a task
cannot vendor, plus native GroupNorm). The baseline is a standalone Triton rewrite of the two kernels
PyTorch Inductor generates for it under `torch.compile(mode="max-autotune")` (pytorch/pytorch v2.15.0,
commit `d0f2af8b`), without `torch._inductor` imports:

- `_triton_tem_fused_addmm_add_mul_sigmoid_0`: the `triton_mm.py.jinja` template rendered for
  `addmm(bias, x, weight.t())` (N, K and the strides are compile-time constants, the batch is a runtime
  argument), with the addmm epilogue and the fused `sigmoid(t) * t + extra_bias` store suffix, the
  `mm_grid` launch grid and `allow_tf32=False` written as `input_precision="ieee"`.
- `_triton_per_fused_native_group_norm_1`: Inductor's persistent reduction for the decomposed
  `native_group_norm`, 32 groups of 64 features per program: the mean as sum / 64, the biased variance as
  the sum of squared deviations / 64, `rsqrt(var + eps)` and the affine. It normalizes the first
  kernel's output in place in `out`, so the baseline allocates nothing.

Inductor autotunes the GEMM over its ROCm config list; without a GPU to autotune on, the baseline fixes
one entry of `ROCmConfigHeuristic.mm_configs` per batch bucket (batch <= 256: 64 x 64 x 64 tiles, 4 warps,
GROUP_M 4; <= 2048: 64 x 128 x 32, 8 warps, GROUP_M 8; larger: 128 x 128 x 32, 8 warps, GROUP_M 16; all
with Inductor's ROCm default of 2 stages and `matrix_instr_nonkdim=16`). The GEMM indexes in int32, as
Inductor does when every operand has fewer than 2^31 elements (true up to the declared batch maximum).
The group-norm kernel likewise fixes one of Inductor's ROCm persistent-reduction configs (`XBLOCK` 32,
with the 4 warps its heuristic gives on wave64); its two-pass statistics are what Inductor emits for
`var_mean` in a persistent reduction (`welford_reduce_fallback`).
`baseline/LICENSE` is the PyTorch BSD-3-Clause license.

## GPU risks (not yet run)

- The kernels were written and reviewed without a GPU or a Triton install; compilation, the batch-bucket
  configs and timing are unverified. The configs were not autotuned on MI355X, so the GEMM may be well
  below the float32 matrix peak.
- The 130 dB threshold comes from CPU emulation of the MFMA accumulation order; if the GPU baseline's
  SQNR is much lower than the modelled 142 dB (for example from MFMA internals or approximate
  `exp`/`rsqrt`), the threshold needs recalibration from the audit's data.
- The 65536-row case writes 1 GiB of output; its float64 reference and the per-sample comparison need
  several GiB of device temporaries.
