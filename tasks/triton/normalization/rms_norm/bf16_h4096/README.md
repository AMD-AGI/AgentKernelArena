# RMSNorm, bfloat16, hidden size 4096

Normalizes each token row by the reciprocal root mean square of its values and scales it by `weight`:
`out = x * rsqrt(mean(x^2) + epsilon) * weight`, with the statistics in float32 and the result rounded to
bfloat16. Sibling of the hidden-size-8192 task; the reference is the same float64 PyTorch computation.

Baseline: the AITER Triton RMSNorm forward (`aiter/ops/triton/normalization/rmsnorm.py` and its device kernels),
including its dispatch: a persistent kernel over rows (one block per row for hidden sizes up to the block
width, which is the case for 4096) and a tiled large-M small-N kernel for more than 8192 rows with at most 2048
columns (not reached by this scenario, kept for fidelity). The quantizing, fused-add and backward variants are
dropped, the `get_num_sms` helper is inlined, and the kernel's inverse-RMS (`rsigma`) scratch output is
allocated inside `run()` since the operator does not return it.

The old tasks merged here (the AITER Triton task, a CK `rmsnorm2d` oracle task and a ROCm-bench kernel with a
zero-centered-gamma flag) are the same operator; this scenario uses the plain gamma of the AITER kernel and
bfloat16 only.

Workloads: token counts at hidden size 4096. Small batches run faster than the 10 us timing floor and are
correctness-only; the performance cases (2560 to 262144 tokens, about 100x) are prefill chunks and batched
prefill. Each token moves 16 KB, so 2560 tokens is the smallest shape expected above the floor. The weight
distribution (uniform 0.5 to 1.5) stands in for trained parameters.
