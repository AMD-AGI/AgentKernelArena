# Batched GEMM, float32, m=256 k=512 n=512

Operator: `out[q] = a[q] @ b[q]` (`torch.bmm`) for `a` [batch, 256, 512] and `b` [batch, 512, 512], all
row-major float32, output float32. The products and the accumulation are IEEE float32: TF32, bf16x3 or
other reduced-precision inputs are a different operator (PyTorch keeps `allow_tf32` off by default).

**Origin.** KernelBench level 1 problem 3 (batched matrix multiplication), the former
`torch2hip/kernelbench/level1/l1n3_Batched_matrix_multiplication` task. Its five cases scaled the batch and
all three inner sizes together, from (b, m, k, n) = (16, 128, 256, 256) to (128, 512, 512, 512). A
scenario fixes the inner sizes, so this task takes the (m, k, n) = (256, 512, 512) case (its `case_2`,
batch 64) and varies only the batch.

**Workloads.** Eight performance batches from 16 to 2048 (128x span; 2.1 to 275 GFLOP). The smallest
launches 128 tiles of 128 x 128, each a 512-deep float32 dot, so it stays well above the 10 us floor even at
the float32 matrix peak. Batches 1, 3 and 100 are correctness-only boundary shapes. Inputs keep KernelBench's
`torch.rand` distribution, uniform on [0, 1).

**Reference.** `torch.bmm` in float64.

**Comparison: `baseline_relative` with `k: 8`, `max_violation_fraction: 1.0e-4` (an override).** The output is
float32, so the baseline's error is the rounding of its 512-deep float32 accumulation (about 4x the RMS of one
output rounding: 136 dB SQNR in simulation), not one output rounding. Any other legitimate summation order has
an independent error of the same size, and the default bound (`k = 2`, no violations) rejects it. A CPU
simulation against the float64 reference (batch 4, 524,288 elements; baseline modelled as the MFMA 16x16x4
order, four-term dots added to a float32 accumulator in k order) gave, for correct float32 orderings: two-term
steps (the 32x32x2 MFMA instruction) 7.1% of elements out of the default bound, a k-sequential scalar loop
19%, PyTorch's CPU float32 `bmm` 0.2%, 2-way split-K 0.001%. With `k = 8` the worst of them (the scalar loop
with separately rounded products, a pessimistic model of an FMA loop) had 0.005% of elements out of bound and
every other ordering none; 16-term steps and 4- and 8-way split-K pass even the default bound. The same bound
still rejects every precision downgrade and control tried: TF32-rounded inputs (92% out of bound), bfloat16
inputs (99%), the fp8 output rounding (100%), dropping one k term (100%), and, through the bias test, the 4-
and 16-ulp output shifts at every batch. bf16x3 split products (three bfloat16 products of round-to-nearest
hi/lo splits) are about as accurate as the scalar loop on these all-positive inputs and pass the elementwise
bound; only their small systematic bias rejects them, through the bias test on the larger batches (at batch
256 their mean error is 1.1e-3 of its RMS against an allowance of 8.6e-4), not on batches up to 64. The
allowance of 1e-4 covers the scalar-loop case with a 2x margin (an independent re-simulation at batches 4 to
256 measured 2.2e-5 to 3.6e-5). Unwritten (NaN) elements are never excused, and the boundary cases keep batch-indexing errors
visible: one wrong product in `b100` is 1% of the elements.

**Baseline.** No production Triton kernel to copy exists for this operator: eager `torch.bmm` in float32
dispatches to a vendor BLAS library, which a task cannot vendor. The baseline is therefore a standalone Triton
rewrite of the kernel that PyTorch Inductor generates for `torch.bmm` under max-autotune (pytorch/pytorch
v2.15.0, commit `d0f2af8b`): the `triton_bmm.py.jinja` template rendered for this layout (M, N, K and the
strides are compile-time constants, the batch is a runtime argument), Inductor's store suffix, the `bmm_grid`
launch grid, `allow_tf32=False` written as `input_precision="ieee"`. Inductor autotunes over its ROCm config
list; without a GPU to autotune on, the baseline fixes one entry of that list (128 x 128 x 32 tiles, 8 warps,
2 stages, GROUP_M 16, `matrix_instr_nonkdim=16`). It indexes in int32, as Inductor does when every operand
has fewer than 2^31 elements (always true up to the declared batch maximum of 4096). `baseline/LICENSE` is the
PyTorch BSD-3-Clause license.

**GPU risks (not yet run).** The fixed config was not autotuned on MI355X, so the baseline may be well below
the float32 matrix peak. The 2048-batch case moves 4 GB of operands and its float64 reference and comparison
need several GB of device temporaries. The 16-ulp output-shift control is rejected by the bias test, not
elementwise; that relies on the MI355X float32 accumulation rounding to nearest (unbiased), as modelled.
