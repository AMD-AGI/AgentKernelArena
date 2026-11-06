# Mean over one axis, float16, N = 256, K = 64

`out[m, k] = mean_n x[m, n, k]` for `x` of shape `(M, 256, 64)`: `torch.mean(x, dim=1)` with float32
accumulation and a float16 result. `M` varies; the reduced axis is strided by `K` elements, so reads
across neighbouring outputs are coalescable by a good kernel.

Origin: vLLM's batch-invariant Triton `mean_kernel` (`vllm/model_executor/layers/batch_invariant.py`),
which the original task wrapped as `mean_dim`. The original cases (`(32,128)`, `(64,256,64)`,
`(16,512)`, `(8,1024,32)` and `(128,64)` over dim 0, all float16) are tiny; this scenario takes the
`(64, 256, 64)` case and scales `M` so the timed cases move 33 MB to 3.4 GB. Smaller `M` is
correctness-only.

Baseline: the vLLM kernel with one program per output element, unchanged except that loaded values are
converted to float32 before the sum (upstream is used with float32 inputs; without the cast a
float16 sum would be tree-reduced in float16 and loosen the baseline-relative bound). The `run()` wrapper
reproduces the launch geometry of `mean_dim`.

Bound: the default `baseline_relative` rule.
