# Jagged-dense batched matmul with broadcast bias, bfloat16, K = N = 256

`out[r, :] = jagged[r, :] @ dense[g] + bias[g]` for every row `r` of group `g`, where the rows of `jagged`
are packed group after group and `seq_offsets[g]:seq_offsets[g + 1]` are the rows of group `g` (the HSTU
jagged-dense BMM of Meta's generative-recommenders). Inputs and output are bfloat16 with float32 accumulation;
the sum is rounded to bfloat16 once, after the bias. There are 64 groups; `K = N = 256`. `total_rows` and
`max_seq_len` (the longest group, passed by the host as in production) vary.

- **Merged sources.** `triton2flydsl/generative_recommenders/jagged_dense_bmm_broadcast_add` and
  `torch2flydsl/jagged_dense_bmm_kernel` implement the same operator. The Triton task also had an
  `elementwise` variant (a bias per row instead of per group); it is dropped, as the plan says. The FlyDSL
  task stores `dense` pre-transposed as `[B, N, K]`; this task keeps the Triton layout `dense[g]` of
  `[K, N]`, which is what the HSTU kernel consumes.
- **Other shapes.** The old Triton suite also swept `K` and `N` (64 to 320, e.g. `K = 192`, `N = 320`) in the same kernel; in schema v3 `K` and `N` are constants of a scenario, and this task covers the production HSTU width `K = N = 256` (the old FlyDSL suite's width). Other widths would be sibling scenarios.
- **Inputs.** `jagged` is standard normal, `dense` has a standard deviation of 1/16 (`1/sqrt(K)`) so the outputs
  stay of order one, and `bias` is standard normal. `inputs.py` generates `seq_offsets`: random group
  lengths summing to `total_rows` with about one group in six empty (the old per-group row lists included
  empty and unaligned groups), one group holding exactly `max_seq_len` rows, and no group longer. Each case
  must satisfy `max_seq_len <= total_rows <= 64 * max_seq_len`; the `max_seq_len` scalar always equals the axis.
- **Baseline.** The forward Triton kernel `jagged_dense_bmm_broadcast_add_kernel` of generative-recommenders,
  via the standalone extraction in the old task (whose upstream commit was not recorded; the provenance
  names the arena revision it was taken from). Changes: the `triton.autotune` decorator and its helper
  functions are removed, since the autotune key collapses to a constant here and tuning inside the first
  (possibly graph-captured) call is not reproducible; one tile configuration from the old tuning space is fixed
  instead (`BLOCK_M = 128`, `BLOCK_N = 128`, `BLOCK_K = 64`, 8 warps, 3 stages), the elementwise variant and
  the `HAS_BIAS` / `ALLOW_TF32` flags are removed, and the wrapper writes into the preallocated output.
  The kernel grid is `(N / BLOCK_N, ceil(max_seq_len / BLOCK_M), n_groups)`; programs past a group's end exit.
- **Comparison.** Default `baseline_relative`; no override.
- **Workloads.** 32K to 2M packed rows (64x span; with the longest group growing from 1K to 64K tokens
  the product of the var axes spans 1000x). At 1 KB moved per row, 32K rows is about 11 us at bandwidth
  speed, so smaller batches (the old suite had only 64 to 1000 rows) are correctness-only: row counts
  1, 48, 333, 1000, 5000 and the all-ones case with 64 rows, including a longest group of 129 (just past one
  128-row tile) and most groups empty.
- **Risk.** Timing of the smallest performance case is near the 10 us floor. The 2M-row case holds about 1 GB of
  `jagged`, 1 GB of `out` and a float64 reference of about 4 GB.
