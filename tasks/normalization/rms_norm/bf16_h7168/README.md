# RMSNorm, bfloat16, hidden size 7168 (FlyDSL baseline)

Normalizes each token row: `out = x * rsqrt(mean(x^2) + 1e-5) * weight`, statistics in float32, result rounded
to bfloat16. Hidden size 7168 is the width of DeepSeek-V3 and Kimi class models. The epsilon is the constant
of the baseline kernel and is not an input. The reference computes in float64 over the bfloat16 inputs.

Baseline: the FlyDSL RMSNorm kernel (ROCm/FlyDSL, `kernels/rmsnorm_kernel.py` at the recorded revision), one
thread block per row; the vectorised path loads the row once into registers, reduces the sum of squares, and
writes the normalized row. The old task carried float16 shapes at 2048 to 8192 and many other builders (large-M
small-N, fused-add, dynamic and smooth quantization); only the plain builder is kept. The vectorised fast path
applies only when the hidden size is a multiple of `BLOCK_THREADS * 8`; with the upstream 256 threads, 7168 is
not (7168 / 2048 = 3.5) and would fall to the scalar generic path. The baseline therefore uses 128 threads per
block (7 tiles per row; reduction slots follow from the wavefront size, nothing else changes). The generic path
remains in the file, unused. `flydsl.expr.buffer_ops` is not imported (it is absent from FlyDSL 0.3.2); the
helper module keeps only the two functions the kernel uses.

The Unsloth `fast_rms_layernorm` task folded into this plan is the same forward operator (its Gemma variant
and backward pass are not part of this task).

Workloads: token counts at hidden size 7168. Small batches run faster than the 10 us timing floor and are
correctness-only; the performance cases (2048 to 204800 tokens, 100x) are prefill chunks and batched prefill.
Each token moves 28 KB. The weight distribution (uniform 0.5 to 1.5) stands in for trained parameters.
