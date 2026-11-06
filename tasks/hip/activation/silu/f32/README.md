# SiLU, float32

`out = x * sigmoid(x)` elementwise over a flattened float32 tensor. The scenario is the float32 path
of the `SiLU` operator of the earlier hip2hip task; `numel` is the single var axis.

**Baseline.** The HIP kernel of that task (float4 vectorized kernel when both pointers are 16-byte
aligned and `numel % 4 == 0`, otherwise a scalar grid-stride kernel, 256 threads, grid capped at
65535 blocks), vendored from this repository at the recorded revision. It was reduced to float32 and
its pybind entry point became the one-op torch library registration written into the preallocated
output. The arithmetic (`x * (1 / (1 + exp(-x)))`) is unchanged.

**Reference.** float64 `x * sigmoid(x)`, evaluated in chunks to bound memory.

**Workloads.** The old shapes stopped at 32x256x128x128 (134 Mi elements, 512 MiB), which is kept.
The old small shapes (1024 ... 32768 elements) run far below the 10 us timing floor and are now
correctness-only, together with odd and unaligned counts that exercise the scalar fallback
(`numel % 4 != 0`). Timed cases run from 5 Mi to 512 Mi elements so the range spans 100x; this span
is set by the framework rule rather than taken from a single production model. Inputs are
`normal(0, 3)` to cover the saturated tails.

**Comparison.** Default `baseline_relative` bound.
