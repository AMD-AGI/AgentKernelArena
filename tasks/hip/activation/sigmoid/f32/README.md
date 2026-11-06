# Scaled sigmoid, float32

`out = sigmoid(a * x) * max_value`, elementwise over a flattened float32 tensor (defaults `a = 1`,
`max_value = 10`). The scenario is the float32 path of the `Sigmoid` operator of the earlier hip2hip
task; `numel` is the single var axis.

**Baseline.** The HIP kernel of that task (vectorized float4 bulk with scalar head and tail,
grid capped at 24 blocks per compute unit), vendored from this repository at the recorded revision.
It was reduced to float32, its environment-variable tuning knobs were replaced by their defaults,
and its pybind entry point became the one-op torch library registration required by the schema.
The kernel arithmetic is unchanged: `1 / (1 + expf(-a*x)) * max_value` in float32.

**Reference.** float64 `sigmoid(a * x) * max_value`, evaluated in chunks to bound memory.

**Workloads.** The old task used a handful of flattened shapes up to 32x256x128x128 (134 Mi elements);
those are kept (the 4D feature-map sizes appear as flattened counts) and extended to 512 Mi elements
(2 GiB in, 2 GiB out) so the timed range spans 100x. Counts below about 3 Mi elements move under
~27 MB and run below the 10 us timing floor, so they are correctness-only. Two correctness cases and
one timed case use non-default `a`/`max_value`. The 5 Mi to 512 Mi performance span is chosen by the
framework rule, not measured from a single production model; the operator itself is a generic
elementwise activation. Inputs are `normal(0, 3)` so a large share of elements fall in the saturated
tails of the sigmoid.

**Comparison.** Default `baseline_relative` bound. Unaligned counts exercise the scalar head and tail.
