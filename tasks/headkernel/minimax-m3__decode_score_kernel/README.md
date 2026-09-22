# minimax-m3__decode_score_kernel

**MiniMax-M3-MXFP4** head kernel - `decode_score_kernel` (Triton / attention, decode).

| field | value |
|---|---|
| GPU time share | 11.63% |
| empirical roofline | 60.0% |
| optimized roofline | 86% |
| e2e uplift measured | -0.15% (not accepted) |
| device symbol | `_decode_score_kernel` |
| production seam | `sglang.kernels.ops.attention.minimax_sparse.decode.flash_with_topk_idx:flash_decode_with_topk_idx` |
| serving contract | ISL 8192 / OSL 1024 / CONC 64 / TP 8 |
| image | `harbor.crusoe.primus-safe.amd.com/hyperloom-image/sglang:v0.5.17-rocm720-mi35x-profilerfix` |
| owner | chaox |
| info rows | MM-1 |

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

- `_decode_score_kernel` in `source/flash_with_topk_idx.py`
- `flash_decode_with_topk_idx` in `source/flash_with_topk_idx.py`

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

`source/` is the **stock** upstream code, byte-identical to
`ut/baseline_ref/*.orig`. Nothing has been pre-optimized.

## Provenance

Copied MiniMax-M3-MXFP4_decode_score_kernel from `/shared_nfs/zihao/headkernel_ut_0831` on 2026-09-14.
Prior optimization results (`_candidate_best/`, `accepted_overlay/`, tuning sweeps,
patches) were deliberately **not** copied - a benchmark that ships the answer
measures nothing. They remain in the upstream package.
Oracle blobs are hardlinked, not duplicated (1 file(s), 6.50 GB shared with the source package).
This package shipped no README. `ut/meta.json` (oracle policy, tolerance, case
geometries), `ut/selection_validation.json` (proof the two legs resolve to
different code) and `ut/negative_check.json` (proof a corrupted output is
rejected) are what it records instead - read those before trusting a speedup.

**Note.** 1.581x holds only when trivial-topk holds; at serving ISL 8192 it does not. RESOLVED 2026-09-16: the missing ut/timing_geometry.pt was located in the ORIGINAL GEAK capture directory named in the package's own ut/README.md, and its sha256 matches the value ut/meta.json records exactly - so this is the authentic artifact, not a reconstruction. build_suite.py now copies it in via the manifest's extra_files. Correction to an earlier note in this file: _geo() is NOT on the frozen-oracle comparison path - unittest.py completes the recorded-oracle check first and only then reaches the geometry, so the earlier claim that the leg died 'before any comparison' was wrong.
