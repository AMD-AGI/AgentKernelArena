# SiLU-and-mul, float16, d = 3200

`out[t] = silu(x[t, :3200]) * x[t, 3200:]` over `x` of shape `[n_tokens, 6400]` in float16. This
sibling of `bf16_d4096` keeps the correctness coverage of the earlier `triton2flydsl/aiter/fused_silu_mul`
task (float16 input, a width that is not a multiple of the 1024 column tile, so the column mask and
partial tiles are exercised) and the `[1024, 6400]` shape of the earlier `torch2flydsl/silu_and_mul_kernel`
task.

**Baseline.** The same AITER `fused_silu_mul` Triton kernel and tile dispatch as `bf16_d4096`
(MIT; `baseline/LICENSE`); see that task's README. The SiLU intermediate is truncated to float16
in the baseline but not in the float64 reference; the default `baseline_relative` bound absorbs it.

**Workloads.** At 19.2 KB of traffic per token, about 1400 tokens are needed to pass the 10 us floor,
so decode batches are correctness-only. Timed cases run from 2048 to 262144 tokens (128x). Inputs are
`normal(0, 2)`; the reference runs in chunks of 8192 rows into a float64 output (about 6.7 GB at the
largest case).
