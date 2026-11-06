# LayerNorm, float16, hidden size 4096 (FlyDSL baseline)

Subtracts the row mean, divides by the root of the biased row variance plus 1e-5, then applies the affine
`weight` and `bias`. The epsilon is the constant of the baseline kernel and is not an input. The reference
computes in float64 over the float16 inputs.

Baseline: the FlyDSL LayerNorm kernel (ROCm/FlyDSL, `kernels/layernorm_kernel.py` at the recorded revision),
one thread block per row. This scenario carries the float16 path of the previous task (the bfloat16 path is
the sibling `bf16_h4096`); the fused-add and quantizing builders are dropped (separate operators). The kernel has a vectorised
fast path that applies only when the hidden size equals `BLOCK_THREADS * 8 * 4`; with the upstream 256 threads
that is 8192, so hidden size 4096 would fall to the scalar generic path. The baseline therefore uses 128
threads per block (reduction slots follow from the wavefront size, nothing else changes), which takes the
fast path for this scenario. The generic path remains in the file, unused.
`flydsl.expr.buffer_ops` is not imported (it is absent from FlyDSL 0.3.2); the helper module keeps only the
two functions the kernel uses.

Workloads: token counts at hidden size 4096. Small batches run faster than the 10 us timing floor and are
correctness-only; the performance cases (2048 to 262144 tokens, 128x) are prefill chunks and batched prefill.
The 4096 x 4096 shape comes from the previous task's shape list. The weight and bias distributions
(uniform 0.5 to 1.5, normal with std 0.1) stand in for trained parameters.
