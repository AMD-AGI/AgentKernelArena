# fp32 matrix-vector product, K = 131072

`y = a @ x` with `a` [m, 131072] float32 and `x` [131072] float32, accumulated in float32 into `y` [m]. The matrix
is read once, so the operator is memory-bound (about 512 MB at m = 1024). The reference is a float64 product in row
chunks of 256. Both inputs are standard normal, so the RMS of `y` is `sqrt(K)` (about 362) and float32 accumulation
order matters (see **Comparison**).

**Origin.** KernelBench level 1, problem 4 (`torch.matmul` of A [M, K] and B [K, 1]). The old task ran five cases
(m = 1024, 2048, 4096 at K = 131072 and m = 2048, 4096 at K = 262144). A v3 scenario fixes K, so this task keeps
K = 131072 (the K = 262144 family is the sibling scenario `k262144`). The vector is declared as [K] rather than [K, 1]
and the output as [m] rather than [m, 1]; the arithmetic is unchanged.

**Workloads.** The KernelBench row counts (1024, 2048, 4096) are included; to reach the required 100x span the
performance cases extend down to m = 64 (32 MB, estimated about 11 us: bytes / 4.5e12 B/s plus the 4 us floor) and up
to m = 8192 (4 GB). Smaller row counts are correctness-only. At small m a good kernel must split K across
programs, which is the design pressure of the task.

**Baseline.** A hand-written Triton split-K kernel (task-original, no upstream code; it replaces the ATen gemv that
the old task used as its PyTorch baseline, so the provenance is the old task's reference module). One program per
(row, K-split) accumulates a 2048-wide fp32 vector across its K segment and reduces it; for fewer than 2048 rows the K
axis is split by up to 32 (partials in a freshly allocated [m, split] buffer, summed by a second kernel). `run(*, a, x, y)` (the output is the declared `y`) only
allocates and launches kernels.

**Comparison: `tolerance` with `atol: 1.122e-3`, `rtol: 0` instead of the default `baseline_relative`.** The error of a
131072-term float32 dot product is accumulation rounding set by the reduction order, not output rounding. Errors below
are in units of `u * sqrt(K)` = 2.16e-5 (`u = 2^-24`; `sqrt(K)` is the RMS of `y`). The rule is shared with the sibling
`k262144`, whose longer float32 chains leave the narrower window and set it: `atol = 52 u sqrt(K)`, and the comparator
adds one output rounding `u |y|`.

- *Why the default fails.* The baseline is an unusually short-chained tree (2048 lane accumulators of at most 64 FMAs,
  split-K below 2048 rows); its RMS error is 1.3 to 2.9. The runtime's own float32 GEMV (`torch.mv`, rocBLAS sgemv on
  MI355X, from 100 rows up) has 7.0: a CPU emulation of one 256-thread block per row, each thread a sequential chain of
  K/256 = 512 products followed by a block tree, reproduces the QA audit's float32 probe under the default bound
  (violation fractions 0.30, 0.30, 0.30, 0.43, 0.36 at m = 8192, 4096, 2048, 1000, 1024 against the audited 0.30,
  0.29, 0.29, 0.42, 0.37). The default bound (twice the baseline's error per element) rejected that probe in 11 of 15
  cases (from 100 rows up violation fraction 0.29 to 0.52 and peak 3.7x to 5.0x; 1.9x on the single row of `m1`; one
  element at 1.07x on `m64`), and in emulation it also rejects the baseline's own kernel with another split-K factor
  (4 % to 12 % of the elements from 32 rows up).
- *Why not `sqnr` or `k` / `max_violation_fraction`.* A correct design's error does not shrink with `|y|`, so on the
  one-row case `m1` both the whole-output SQNR and a baseline-relative bound (one baseline error sample) are ratios of
  two independent normal draws: a design as accurate as the GPU GEMV falls below a 121.5 dB SQNR threshold in 29 % of
  one-row draws, and under `k: 8, max_violation_fraction: 0.01` the emulated GPU GEMV still fails 7 of 30 case draws.
  K and the standard-normal inputs are fixed by the task, so the error scale is fixed too and an absolute bound holds
  for every row count.
- *Calibration.* CPU emulations of float32 reduction orders (every add rounded to nearest as IEEE float32 does, FMAs
  rounded once) on this task's input distributions, one full correctness run (two draws of all 15 cases, 34,978
  elements):

  | Implementation | RMS error | largest error / bound |
  | --- | --- | --- |
  | the baseline's arithmetic | 1.3 to 2.9 | 0.25 |
  | one Triton program per row, 4096 or 8192 lanes | 2.6 | 0.23 |
  | 1024 threads or 1024 Triton lanes per row (128-term chains) | 3.7 | 0.33 |
  | GPU GEMV model (256 threads, 512-term chains), with or without FMA | 7.0 | 0.61 |
  | one 64-lane wavefront per row, one accumulator per lane (2048-term chains) | 13.6 | 1.05 (fails) |
  | CPU MKL `sgemv` | 54 | 5.5 (fails) |
  | `shifted_16_ulp` control | 16, proportional to `y` | 1.24 (fails) |

  In 196,608 further rows of the GPU GEMV model the error was Gaussian (RMS 6.97, kurtosis 3.01) and at most 31.5
  (4.5 sigma), 0.61 of the bound; a full correctness run of a design that accurate in every case exceeds the bound
  with probability about 1e-9. The audit's probe rows at 64 rows and fewer (rocBLAS's split-K path) were within 1.9x
  of the old bound, which is a few units there (twice the baseline's error plus one rounding), far inside the new one.
- *Controls.* `shifted_16_ulp` moves every element by `16 u |y|`, which exceeds the bound where `|y| > 3.5 sqrt(K)`:
  a control run (two draws of every case) is expected to put 23 elements over the bound, mostly in the cases of 2048
  rows and more, so it survives with probability below 1e-10. The emulated run had 24, the first at `m100`, each
  1.01x to 1.24x the bound. The shift is only 2.3 times the GPU GEMV's RMS error, so no bound separates them by a
  wider amplitude; a design with longer float32 chains (the wavefront-per-row row above, SQNR 122 dB against the
  shift's 120.3 dB) is as inaccurate as the shift and is rejected. `fp8_rounded` moves `y` by up to 6 % (about 10 for
  a typical element, 10^4 times `atol`; at least 80x the bound on `m1` in emulation). `noop` (NaN-poisoned
  output), `zeros` and `half_written` (errors of order `|y|`), `stale_outputs` (the first draw's output returned for
  the second) and `input_write` (caught by the input snapshot) fail independently of the bound.
- *Risk.* The bound is calibrated with CPU emulation, not on the GPU; the GPU GEMV's error scale is inferred from the
  matching violation fractions. `atol` assumes the declared standard-normal inputs and K.

**GPU audit risks.** Lowest performance case (m = 64) sits near the 10 us floor and the baseline's split-K choice is
untuned; if the audit measures it below 10 us, raise the minimum performance row count (and the maximum, to keep the
span). The 4 GB input at m = 8192 plus chunked float64 reference stays well under the memory budget.
