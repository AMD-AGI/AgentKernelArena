# Scaled tanh, float32

`out = tanh(a * x) * scale`, elementwise over a flat float32 tensor (defaults `a = 1`, `scale = 10`),
the scalar-parameterised tanh of the retired `hip2hip/TanH` task. The reference is float64.

**Baseline.** The HIP kernel of the old task's provided baseline (`hip_11178_TanH_ref.hip` at
AgentKernelArena commit 1464c935, Apache-2.0): a vectorised grid-stride kernel with a float4/float2
body and scalar head/tail chosen from pointer alignment. Only the float32 path is kept, and the
entry is registered as the arena `run` op (no pybind) writing into the preallocated output; the
kernel itself is unchanged.

**Workloads.** Element counts are those of elementwise activation/feature-map tensors (the old task
used 1K to 134M elements, we extend to 512M so the performance cases span 128x). Cases under about
3.4M elements (the 10 us floor at 8 bytes per element) are correctness-only. Extra correctness cases
cover a single element, an odd tail-only length, a non-multiple of 16, an unaligned large length,
and non-default `a`/`scale` values (including negative) so a hard-coded `a = 1, scale = 10` fails.

**Comparison.** Default `baseline_relative`: a candidate must be about as accurate as the baseline's
`tanhf`, so fast approximate tanh is rejected.

**Risks.** The largest case moves 4 GiB; the reference needs about 20 GB of float64 temporaries.
