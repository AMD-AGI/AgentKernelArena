# minimax-m3__gqa_share_sparse_fwd_kernel

**MiniMax-M3-MXFP4** head kernel - `gqa_share_sparse_fwd_kernel` (triton / attention, prefill).

| field | value |
|---|---|
| GPU time share | 3.98% |
| empirical roofline | 10.72% |
| optimized roofline | 14.1% |
| e2e uplift measured | 10.28% (GEAK self-measured) |
| device symbol | `_gqa_share_sparse_fwd_kernel` |
| production seam | `sglang.kernels.ops.attention.minimax_sparse.prefill.topk_sparse:flash_prefill_with_gqa_share_sparse` |
| serving contract | ISL 8192 / OSL 1024 / CONC 64 / TP 8 |
| image | `harbor.crusoe.primus-safe.amd.com/hyperloom-image/sglang:v0.5.17-rocm720-mi35x-profilerfix` |
| owner | chaox |
| info rows | MM-5 |

## Layout

```
config.yaml              arena task schema + a headkernel: provenance block
scripts/task_runner.py   compile | correctness | performance
scripts/_bench.py        native 10 warmup / 100 measured timing
source/                  THE EDITABLE KERNEL - change only this
ut/                      frozen GEAK op package (oracle, harness, overlays)
ut/kernel_src/           symlinks back into source/ - same bytes, two views
```

Edit targets:

- `_gqa_share_sparse_fwd_kernel` in `source/topk_sparse.py`
- `flash_prefill_with_gqa_share_sparse` in `source/topk_sparse.py`

## Running it

On one GPU from the optimization pool (never the serving set), inside the image above:

```bash
cd <task>
python3 scripts/task_runner.py compile
python3 scripts/task_runner.py correctness
python3 scripts/task_runner.py performance
```

- **compile** AST-parses `source/` and asserts every target symbol is defined there.
  No GPU needed.
- **correctness** runs `ut/unittest.py`: the frozen live-capture oracle
  (`ut/reference_io.pt`) plus random-value parity against the live baseline leg,
  at tol `0.02`. Exit 0 pass, 1 correctness fail, 2 environment,
  3 harness incomplete.
- **performance** replays the captured argument records from `ut/reference_io.pt`
  with 10 warmup + 100 measured iterations and reports the mean cuda-event device
  time. If a record cannot be rebuilt it falls back to the GEAK interleaved
  median-of-3 legs and says so in `build/performance_report.json`.
  A run whose unit test did not pass reports no cases at all.

## Starting point

`source/` is seeded from the **stock** pre-optimization code
(`baseline_ref/*.orig` in the upstream package). The package itself shipped a
kernel_src that a previous GEAK run had already tuned (1 file(s)); that version is kept out of the task at
`_prior_solutions/minimax-m3__gqa_share_sparse_fwd_kernel/` so this benchmark starts where every other task in
the suite starts.

## Provenance

Copied MiniMax-M3-MXFP4_gqa_share_sparse_fwd_kernel from `/shared_nfs/zihao/headkernel_ut_0831` on 2026-09-14.
Prior optimization results (`_candidate_best/`, `accepted_overlay/`, tuning sweeps,
patches) were deliberately **not** copied - a benchmark that ships the answer
measures nothing. They remain in the upstream package.
Oracle blobs are hardlinked, not duplicated (1 file(s), 7.28 GB shared with the source package).
The original package README is preserved at `ut/README.md` and is the authority on
this op's measurement caveats - read it before trusting a speedup.

**Note.** RESOLVED 2026-09-16: the missing ut/timing_geometry.pt was located in the ORIGINAL GEAK capture directory named in the package's own ut/README.md, and its sha256 matches the value ut/meta.json records exactly - so this is the authentic artifact, not a reconstruction. build_suite.py now copies it in via the manifest's extra_files. Correction to an earlier note in this file: _geo() is NOT on the frozen-oracle comparison path - unittest.py completes the recorded-oracle check first and only then reaches the geometry, so the earlier claim that the leg died 'before any comparison' was wrong.
