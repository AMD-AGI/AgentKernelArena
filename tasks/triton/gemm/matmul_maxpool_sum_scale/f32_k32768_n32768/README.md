# Matmul + max pool + sum + scale, float32, in = out = 32768

Operator: `out = scale_factor * sum(max_pool1d(x @ weight.T + bias, kernel_size=2), dim=1)` for `x`
[batch, 32768], an `nn.Linear(32768, 32768)` weight [32768, 32768] and bias [32768], all row-major float32,
`scale_factor = 0.5`, output float32 [batch]. The product and its accumulation are IEEE float32 (TF32, bf16x1
or other reduced-precision inputs are a different operator; PyTorch keeps `allow_tf32` off by default). The
pool takes the maximum of each non-overlapping pair of output features (2j, 2j + 1), stride 2, no padding;
each row's 16384 pair maxima are summed and scaled.

**Origin.** KernelBench level 2 problem 55 (`Matmul_MaxPool_Sum_Scale`), the former
`torch2hip/kernelbench/level2/l2n55_Matmul_MaxPool_Sum_Scale` task, whose module is built with
`get_init_inputs() = [32768, 32768, 2, 0.5]` and whose five cases escalated the batch from 32 to 512. The pool
size is part of the scenario; `scale_factor` is the module's scalar argument, kept at its value 0.5.

**Workloads.** Eight performance batches from 4 to 512 (128x span): the original five (32 to 512) plus 4, 8
and 16 to reach the required span. Every call reads the 4 GiB weight, so even batch 4 takes about a
millisecond; batch 512 is a 1.1 TFLOP float32 product. Batches 1, 3, 100 and 300 are correctness-only
boundary shapes, one in each block-size bucket of the baseline (1 and 3 below 32 rows, 100 in the 64-row
bucket, 300 a partial 128-row tile). Inputs keep KernelBench's distributions: `x` is `torch.rand`, uniform on
[0, 1); weight and bias follow `nn.Linear`'s default initialization, uniform on +-1/sqrt(32768).

**Reference.** `addmm`, `max_pool1d` and the sum in float64, in 4096-feature chunks to bound the float64
temporaries (a 32768 x 32768 float64 weight would be 8 GiB).

**Comparison: `sqnr` with `min_db: 122` (an override).** Each output element is a 16384-term float32 sum (of
maxima of 32768-deep float32 dot products), so its error depends on the summation order far more than on one
output rounding, and the output of a small batch has only a few elements. A CPU simulation against the
float64 reference (32 rows at the full K = N = 32768; the baseline modelled as 4-term MFMA steps added to a
float32 accumulator in k order followed by Inductor's reduction, 2048 lanes of 8 sequential adds and a tree)
gave the baseline 4.1e-8 relative RMS error (147.7 dB), an output-rounding-level error. Correct float32
implementations differ by up to an order of magnitude: one 16384-term tree 147.7 dB, a float64 accumulator
150.6 dB, per-tile tree partials added one after another (as a fused GEMM epilogue with atomics does) 138.2,
135.6, 132.8, 129.8, 126.2 and 123.6 dB for 64, 128, 256, 512, 1024 and 2048 tiles (2.3e-7 relative RMS for 256
tiles), 2-term MFMA steps or a scalar FMA loop for the product 146.6 and 147.3 dB with Inductor's reduction and
129.2 to 132.7 dB with 256 or 512 atomic tiles, PyTorch's CPU float32 product 150.0 dB with a tree, and a fully
sequential single-thread sum 113.8 dB (2.0e-6). The default bound rejects every tile-partial order, and even
`k = 8` with a 1% violation allowance rejects the 256-, 512-, 1024- and 2048-tile sums (6% to 44% of the
elements out of bound), so it would reject the natural fused design. SQNR >= 122 dB (a relative RMS error of at
most 7.9e-7, about 13 units of float32 roundoff) accepts every parallel order tried and rejects the fully
sequential sum, which is 50 times less accurate than the baseline. It rejects the framework's 16-ulp output
shift (120.6 dB; an 8-ulp shift, 126.6 dB, is within the spread of correct orders and passes) and every precision
downgrade and wrong variant tried: TF32-rounded inputs (107.9 dB), bfloat16 inputs (88.2 dB), half-precision
tile partials (111.2 dB), the bias left out (75.4 dB), the last pair dropped (86.1 dB), pairs (j, j + 16384)
instead of (2j, 2j + 1) (48.1 dB), an average instead of a max pool (0.1 dB), fp8 output rounding (2.9 dB), and
half-written or unwritten outputs. Non-finite elements are never excused. With one element, the `b1` case judges
a single sum, so its pass rate is a per-draw tail probability: below 0.1% failures for the 256-tile atomic order,
about 1.5% for 512 tiles, about 10% for 1024 tiles and 25% for 2048 tiles (3 and 9 of the 32 simulated rows were
out of bound for 1024 and 2048 tiles); larger batches average the error.

**Baseline.** No production Triton kernel exists for this module, and eager PyTorch dispatches the product to
a vendor BLAS library, which a task cannot vendor. The baseline is therefore a standalone Triton rewrite of
the two kernels PyTorch Inductor generates for the module under `torch.compile(mode="max-autotune")` on ROCm
(pytorch/pytorch v2.15.0, commit `d0f2af8b`), written by hand from Inductor's templates and code generator
because no GPU was available to run Inductor:

1. the `triton_mm.py.jinja` mm template for `x` and the transposed weight view with the `addmm` epilogue (the
   bias is the template's prefix input and is added in the generated store suffix; the post-grad
   `unfuse_bias_add_to_pointwise` pattern does not apply because the product feeds a max pool, not a pointwise
   op); launch grid `mm_grid`, `allow_tf32=False` written as `input_precision="ieee"`;
2. the looped sum reduction with the max pool fused in: `max_pool1d` lowers through
   `max_pool2d_with_indices` to Inductor's low-memory max pool, whose two-element window is unrolled into a
   NaN-propagating `tl.maximum` of two loads, and the product with `scale_factor` is applied to the reduced
   value before the store. Inductor bakes the constant 0.5 in; the baseline takes it as the scalar argument.

Inductor autotunes the template over its ROCm config list; without a GPU to autotune on, the baseline picks one
entry of that list per M bucket (32 x 32 x 128 with 4 warps up to 32 rows, 64 x 64 x 64 with 4 warps up to 256
rows, 128 x 128 x 32 with 8 warps above; 2 stages, `matrix_instr_nonkdim=16`). The reduction uses Inductor's
contiguous inner-reduction config (one row per program, 2048-wide blocks, 8 warps) as a single kernel for every
batch; Inductor may split the reduction of a small batch into two kernels, which only matters for the
microseconds the reduction takes next to the millisecond product. The intermediate product lives in a buffer
allocated on each call, as Inductor's generated wrapper does. Extern (rocBLAS) choices that max-autotune would
also benchmark are not part of a vendorable baseline. `baseline/LICENSE` is the PyTorch BSD-3-Clause license.

**GPU risks (not yet run).** The block configs were not autotuned on MI355X, so the baseline may be well below
the float32 matrix peak; batch 512 is expected to take on the order of 10 ms. Each correctness draw holds a
4 GiB weight and the evaluator keeps several copies (two draws, a working copy, the reference's input clone),
about 20 GiB of device memory; the chunked float64 reference adds about 1 GiB. The simulated MFMA rounding is
a model; the baseline's SQNR (147.7 dB simulated) has 25 dB of margin to the threshold.
