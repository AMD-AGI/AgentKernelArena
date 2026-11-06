# fp32 matrix-vector product, K = 262144

`y = a @ x` with `a` [m, 262144] float32 and `x` [262144] float32, accumulated in float32 into `y` [m]. The matrix
is read once, so the operator is memory-bound (about 1 GB at m = 1024). The reference is a float64 product in row
chunks of 256. Both inputs are standard normal, so the RMS of `y` is `sqrt(K)` = 512 and float32 accumulation order
matters (see **Comparison**).

**Origin.** KernelBench level 1, problem 4 (`torch.matmul` of A [M, K] and B [K, 1]). The old task ran five cases
(m = 1024, 2048, 4096 at K = 131072 and m = 2048, 4096 at K = 262144). A v3 scenario fixes K, so this task keeps
K = 262144 (the K = 131072 family is the sibling scenario `k131072`). The vector is declared as [K] rather than [K, 1]
and the output as [m] rather than [m, 1]; the arithmetic is unchanged.

**Workloads.** The KernelBench row counts (2048, 4096) are included; to reach the required 100x span the performance
cases extend down to m = 32 (32 MB, estimated about 11.5 us: bytes / 4.5e12 B/s plus the 4 us floor) and up to
m = 4096 (4 GB). Smaller row counts are correctness-only. At small m a good kernel must split K across programs, which
is the design pressure of the task.

**Baseline.** A hand-written Triton split-K kernel (task-original, no upstream code; it replaces the ATen gemv that
the old task used as its PyTorch baseline, so the provenance is the old task's reference module). One program per
(row, K-split) accumulates a 2048-wide fp32 vector across its K segment and reduces it; for fewer than 2048 rows the K
axis is split by up to 32 (partials in a freshly allocated [m, split] buffer, summed by a second kernel). `run(*, a, x, y)` (the output is the declared `y`) only
allocates and launches kernels.

**Comparison: `tolerance` with `atol: 1.587e-3`, `rtol: 0` instead of the default `baseline_relative`.** The error of a
262144-term float32 dot product is accumulation rounding set by the reduction order, not output rounding. Errors below
are in units of `u * sqrt(K)` = 2^-15 = 3.05e-5 (`u = 2^-24`; `sqrt(K)` is the RMS of `y`). The rule is shared with the
sibling `k131072`: `atol = 52 u sqrt(K)`, and the comparator adds one output rounding `u |y|`.

- *Why the default fails.* The baseline is an unusually short-chained tree (2048 lane accumulators of at most 128
  FMAs, split-K below 2048 rows); its RMS error is 1.6 to 3.8. The runtime's own float32 GEMV (`torch.mv`, rocBLAS
  sgemv on MI355X, from 256 rows up) has 9.7: a CPU emulation of one 256-thread block per row, each thread a
  sequential chain of K/256 = 1024 products followed by a block tree, reproduces the QA audit's float32 probe under the
  default bound (violation fractions 0.34, 0.34, 0.46, 0.50, 0.49, 0.56 at m = 4096, 2048, 1024, 1000, 512, 256
  against the audited 0.34, 0.33, 0.47, 0.50, 0.49, 0.55). The default bound (twice the baseline's error per element)
  rejected that probe in 9 of 15 cases (from 256 rows up violation fraction 0.33 to 0.55 and peak 4.0x to 5.5x; on
  rocBLAS's split-K path one of three elements at 1.72x on `m3`, 5 % at 1.20x on `m100`, 2.3 % at 1.41x on `m128`),
  and in emulation it also rejects the baseline's own kernel with another split-K factor (6 % to 25 % of the elements
  from 32 rows up).
- *Why not `sqnr` or `k` / `max_violation_fraction`.* A correct design's error does not shrink with `|y|`, so on the
  one-row case `m1` both the whole-output SQNR and a baseline-relative bound (one baseline error sample) are ratios of
  two independent normal draws: a design as accurate as the GPU GEMV falls below a 121.5 dB SQNR threshold in 39 % of
  one-row draws, and under `k: 8, max_violation_fraction: 0.01` the emulated GPU GEMV still fails 18 of 30 case draws.
  K and the standard-normal inputs are fixed by the task, so the error scale is fixed too and an absolute bound holds
  for every row count.
- *Calibration.* CPU emulations of float32 reduction orders (every add rounded to nearest as IEEE float32 does, FMAs
  rounded once) on this task's input distributions, one full correctness run (two draws of all 15 cases, 18,610
  elements):

  | Implementation | RMS error | largest error / bound |
  | --- | --- | --- |
  | the baseline's arithmetic | 1.6 to 3.8 | 0.29 |
  | one Triton program per row, 4096 or 8192 lanes | 2.8 to 3.1 | 0.23 |
  | 1024 threads or 1024 Triton lanes per row (256-term chains) | 5.1 | 0.42 |
  | GPU GEMV model (256 threads, 1024-term chains), with or without FMA | 9.7 | 0.85 |
  | one 64-lane wavefront per row, one accumulator per lane (4096-term chains) | 19.4 | 1.62 (fails) |
  | CPU MKL `sgemv` | 77 | 8.1 (fails) |
  | `shifted_16_ulp` control | 16, proportional to `y` | 1.21 (fails) |

  In 131,072 further rows of the GPU GEMV model the error was Gaussian (RMS 9.72, kurtosis 3.02) and at most 41.9
  (4.3 sigma), 0.80 of the bound; a full correctness run of a design that accurate in every case exceeds the bound
  with probability about 1e-3 (the audit's single-draw probe, which takes that path from 256 rows up, about 5e-4). The
  audit's probe rows at 128 rows and fewer were within 1.72x of the old bound, which is a few units there (twice the
  baseline's error plus one rounding), far inside the new one.
- *Controls.* `shifted_16_ulp` moves every element by `16 u |y|`, which exceeds the bound where `|y| > 3.5 sqrt(K)`:
  a control run (two draws of every case) is expected to put 14 elements over the bound, mostly in the cases of 1000
  rows and more, so it survives with probability below 1e-6. The emulated run had 18, the first at `m128`, each 1.04x
  to 1.21x the bound. The shift is only 1.65 times the GPU GEMV's RMS error, so no bound separates them by a wider
  amplitude, and `atol` sits where both risks are small (52 balances a 1e-3 false rejection of the GPU GEMV against a
  1e-6 survival of the shift; 50 would give 3e-3 and 1e-10, 54 give 3e-4 and 1e-4). A design with longer float32
  chains (the wavefront-per-row row above, SQNR 119 dB against the shift's 120.2 dB) is less accurate than the shift
  and is rejected. `fp8_rounded` moves `y` by up to 6 % (about 15 for a typical element, 10^4 times `atol`; at least
  900x the bound on `m1` in emulation). `noop` (NaN-poisoned output), `zeros` and `half_written` (errors of order
  `|y|`), `stale_outputs` (the first draw's output returned for the second) and `input_write` (caught by the input
  snapshot) fail independently of the bound.
- *Risk.* The bound is calibrated with CPU emulation, not on the GPU; the GPU GEMV's error scale is inferred from the
  matching violation fractions. `atol` assumes the declared standard-normal inputs and K.

**GPU audit risks.** Lowest performance case (m = 32) sits near the 10 us floor and the baseline's split-K choice is
untuned; if the audit measures it below 10 us, raise the minimum performance row count (and the maximum, to keep the
span). The 4 GB input at m = 4096 plus chunked float64 reference stays well under the memory budget.
