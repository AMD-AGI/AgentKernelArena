# Exact GELU, float32, flat tensor

`out = 0.5 * x * (1 + erf(x / sqrt(2)))` (`torch.nn.functional.gelu`, `approximate='none'`) over a
float32 tensor of `numel` elements. Elementwise, so the old tasks' ranks (1D to 4D) are one flat view;
rank variety is not a separate axis. The inputs are uniform on [-6, 6] (the KernelBench input range,
which covers both GELU tails). The reference is float64 and chunked.

- **Merged sources.** `hip2hip/GELU` (gpumode activation suite, 1K to 134M elements) and
  `torch2hip/kernelbench/level1/l1n26_GELU_` (16M to 537M elements; its baseline path is the same
  operator). Their HIP kernels differ only in names, so the gpumode kernel is the baseline.
- **Baseline.** The HIP kernel provided by `hip2hip/GELU`, task-original (not an upstream production
  kernel): scalar kernel up to 4096 elements, otherwise a vectorized float4 kernel with a scalar prologue
  and epilogue for alignment and tail, elements per thread chosen by size, blocks capped at 24 per CU.
  It was written for an in-place clone-then-update launcher; here it reads `x` and writes `out`
  (vector path only when both pointers share 16-byte alignment, else scalar), and the environment-variable
  tuning, the cached CU-count and the double path are removed so the source meets the HIP candidate rules.
  Block caps and tuning constants are unchanged defaults.
- **Workloads.** Six performance cases from 4.2M to 537M elements (128x span). Each moves 8 bytes per
  element, estimated at least 11 us at 4.2M. The 4.2M case (4096 tokens x 1024) is representative rather
  than a verbatim old case, needed to reach the 100x span from the 537M KernelBench shape. Smaller gpumode
  shapes are correctness-only. The largest case uses 2.1 GB each for input and output and a 4.3 GB float64
  reference.
- **Comparison.** Default `baseline_relative`; no override.
- **Risk.** HIP build and timing are unverified on a GPU. Erf-based outputs for large negative inputs are
  tiny, so the relative bound depends on the baseline's own `erff` error; the audit should confirm it.
