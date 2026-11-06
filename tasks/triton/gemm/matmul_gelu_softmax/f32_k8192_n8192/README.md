# Matmul + GELU + softmax, float32, in = out = 8192

Operator: `out = softmax(gelu(x @ weight.T + bias), dim=1)` for `x` [batch, 8192], an `nn.Linear(8192, 8192)`
weight [8192, 8192] and bias [8192], all row-major float32, output float32. The product and its accumulation
are IEEE float32 (TF32, bf16x1 or other reduced-precision inputs are a different operator; PyTorch keeps
`allow_tf32` off by default), the GELU is the exact erf form (`approximate="none"`), and the softmax runs over
the 8192 output features of each row.

**Origin.** KernelBench level 2 problem 99 (`Matmul_GELU_Softmax`), the former
`torch2hip/kernelbench/level2/l2n99_Matmul_GELU_Softmax` task, whose module is built with
`get_init_inputs() = [8192, 8192]` and whose five cases escalated the batch from 256 to 4096. The module has no
dropout or other training-mode behaviour.

**Workloads.** Eight performance batches from 32 to 4096 (128x span): the original five (256 to 4096) plus 32,
64 and 128 to reach the required span. Every call reads the 256 MiB weight, so even batch 32 stays well above
the 10 us floor; batch 4096 is a 0.55 TFLOP float32 product. Batches 1, 3, 100 and 1000 are correctness-only
boundary shapes, one in each block-size bucket of the baseline (1 and 3 below 32 rows, 100 in the 64-row
bucket, 1000 a partial 128-row tile). Inputs keep KernelBench's distributions: `x` is `torch.rand`, uniform on
[0, 1); weight and bias follow `nn.Linear`'s default initialization, uniform on +-1/sqrt(8192).

**Reference.** `addmm`, GELU and softmax in float64.

**Comparison: `sqnr` with `min_db: 122` (an override).** The output error of a correct float32 implementation
is dominated by the rounding of the 8192-deep float32 accumulation of the product (and, second, by the
summation order of the softmax normalizer), not by one output rounding, and it is heteroscedastic: the error of
an element scales with `out * gelu'(y)`, which varies more than tenfold across a row. The default elementwise
bound (`baseline_relative`, `k = 2`) therefore rejects correct implementations, and its bias test is unsuited
to a softmax: a different summation order of the row normalizer shifts a whole row by one common relative
error, which the bias test reads as a systematic bias and compares against the baseline's own, randomly small,
normalizer error. A CPU simulation against the float64 reference (2 rows, 16,384 elements; the baseline
modelled as 4-term MFMA steps added to a float32 accumulator in k order followed by Inductor's online softmax
with 2048 lanes; exp and erf correctly rounded) gave the baseline 132.6 dB SQNR and rejected, under the default
bound, a scalar FMA loop (7.0% of elements out of bound), 2-term MFMA steps (the 32x32x2 instruction, 3.5%),
PyTorch's CPU float32 product (1.0%), 2-way split-K (0.4%) and a normalizer summed from 64 atomically added tile
partials (0.2%); even `k = 8` rejected the scalar loop (0.31%), the 2-term steps (0.13%) and the CPU product
(0.006%), and a sequentially summed normalizer failed the bias test at every `k`. Under SQNR the correct
variants score 124.9 to 140.7 dB: each of {4-term MFMA steps, 2-term steps, scalar FMA loop} combined with each
of {Inductor online softmax, two-pass tree softmax, 64 atomically added tile partials, per-tile online merge,
exp2-based exp with a 1-ulp error and a reciprocal multiply, single-thread sequential normalizer}, plus 2- and
8-way split-K, the CPU product and 256 tile partials, except the most sequential combination (scalar FMA loop
and sequential normalizer, 121.1 dB), which is rejected. The threshold rejects every
precision downgrade and control tried: TF32-rounded inputs (84.3 dB), bfloat16 inputs (66.2 dB), the tanh GELU
approximation (91.1 dB), the framework's 16-ulp output shift (119.9 dB; an 8-ulp shift, 125.1 dB, is within
the error of correct float32 orderings and passes), fp8 output rounding, and half-written or unwritten outputs
(non-finite elements are never excused). SQNR is a whole-tensor measure; the boundary cases keep row-indexing
errors visible, since one wrong row in `b100` is 1% of the signal energy (20 dB) and in `b1000` 0.1% (30 dB).

**Baseline.** No production Triton kernel exists for this module, and eager PyTorch dispatches the product to
a vendor BLAS library, which a task cannot vendor. The baseline is therefore a standalone Triton rewrite of
the two kernels PyTorch Inductor generates for the module under `torch.compile(mode="max-autotune")` on ROCm
(pytorch/pytorch v2.15.0, commit `d0f2af8b`), written by hand from Inductor's templates and code generator
because no GPU was available to run Inductor:

1. the `triton_mm.py.jinja` mm template for `x` and the transposed weight view, with no epilogue: on a GPU the
   post-grad pattern `unfuse_bias_add_to_pointwise` rewrites `addmm` whose users are all pointwise into `mm`
   plus a bias add, so the bias moves into the next kernel; launch grid `mm_grid`, `allow_tf32=False` written
   as `input_precision="ieee"`;
2. the looped inner reduction with the bias add, the erf GELU and the online softmax fused
   (`triton_helpers.online_softmax_combine` / `online_softmax_reduce` copied in; ROCm uses the per-lane vector
   accumulators), the GELU recomputed in the second pass.

Inductor autotunes the template over its ROCm config list; without a GPU to autotune on, the baseline picks one
entry of that list per M bucket (32 x 32 x 128 with 4 warps up to 32 rows, 64 x 64 x 64 with 4 warps up to 256
rows, 128 x 128 x 32 with 8 warps above; 2 stages, `matrix_instr_nonkdim=16`). The reduction uses Inductor's
contiguous inner-reduction config (one row per program, 2048-wide blocks, 8 warps). The intermediate product
lives in a buffer allocated on each call, as Inductor's generated wrapper does. Extern (rocBLAS) choices that
max-autotune would also benchmark are not part of a vendorable baseline. `baseline/LICENSE` is the PyTorch
BSD-3-Clause license.

**GPU risks (not yet run).** The block configs were not autotuned on MI355X, so the baseline may be well below
the float32 matrix peak. The simulated MFMA rounding (each 4-term step added with one rounding) is a model;
if the hardware rounds differently, the baseline's SQNR (132.6 dB simulated) moves, with 10 dB of margin to the
threshold. The largest case's float64 reference needs about 1.6 GB of device temporaries.
