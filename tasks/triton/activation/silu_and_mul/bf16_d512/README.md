# SiLU-and-mul, bfloat16, d = 512

`out[r] = silu(x[r, :512]) * x[r, 512:]` over `x` of shape `[n_tokens, 1024]`, where `n_tokens` counts
(token, expert) rows of a MoE activation. This sibling of `bf16_d4096` covers the tuned `d = 512`
dispatch branch of the AITER baseline (`_pick_block_n` / `_pick_block_m`: 256-wide N tiles up to 4096
rows, one 512-wide tile above), which the earlier `triton2flydsl/aiter/fused_silu_mul` task exercised at
`last = 1024` (32, 2048, 57904 rows).

**Baseline.** The same AITER `fused_silu_mul` Triton kernel and tile dispatch as `bf16_d4096`
(MIT; `baseline/LICENSE`); see that task's README.

**Workloads.** At 3 KB of traffic per row about 9000 rows are needed to pass the 10 us floor, so
decode and small batches (and the dispatch-threshold cases 4096 and 4097) are correctness-only. Timed
cases run from 16384 to 2097152 rows (128x; the largest corresponds to roughly 262144 tokens x top-8
experts). Inputs are `normal(0, 2)`; the reference runs in chunks of 8192 rows into a float64 output
(about 8.6 GB at the largest case).
