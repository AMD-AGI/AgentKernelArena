# LayerNorm, bfloat16, hidden size 6144

Subtracts the row mean, divides by the root of the biased row variance plus `epsilon`, then applies the
affine `weight` and `bias`. The reference computes in float64 over the bfloat16 inputs.

Baseline: AITER's Triton `layer_norm` forward (ROCm/aiter at the recorded revision), one program per row.
The kernel is copied unchanged and the launcher is inlined into `run()`; the per-row `mean` and `rstd`
scratch outputs of the upstream function are allocated inside `run()` and are not part of the interface.
It replaces the Composable Kernel `layernorm2d_fwd` oracle of the previous torch2flydsl task and the
RocmBench layernorm (which also carried backward kernels; only the forward is a task here).

Workloads: token counts at hidden size 6144 (the 6144-wide case of the previous torch2flydsl task; not a power of two, so the kernel's masked tail block is exercised). Decode-sized batches run faster
than the 10 us timing floor and are correctness-only; the performance cases (2048 to 262144 tokens, 128x)
are prefill chunks and batched prefill. The
weight and bias distributions (uniform 0.5 to 1.5, normal with std 0.1) stand in for trained parameters.
Epsilon 1e-5 is the usual LayerNorm default.
