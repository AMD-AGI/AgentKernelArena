# kimi-k3__moe_gemm2_stage2

**Kimi-K3** head kernel - `moe_gemm2_0` (aiter asm (AOT) / flydsl opus - fused_moe_2stages stage-2 down - mxfp4 + bf16, prefill).

| field | value |
|---|---|
| GPU time share | 5.47% |
| empirical roofline | 0.230 |
| optimized roofline | - |
| e2e uplift measured | - |
| device symbol | `moe_gemm2_0` |
| production seam | `aiter.ops.flydsl.moe_kernels:flydsl_moe_stage2` |
| serving contract | ISL 8192 / OSL 1024 / CONC 64 / TP 8 |
| image | `harbor.crusoe.primus-safe.amd.com/hyperloom-image/sglang-rocm-k3:rocm720-mi35x-k3-20260727-tl312-08011830` |
| owner | Hongtao |
| info rows | K3-5 |

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

- `flydsl_moe_stage2` in `source/flydsl/moe_kernels.py`

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
  at tol `0.05`. Exit 0 pass, 1 correctness fail, 2 environment,
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

Copied Kimi-K3_moe_gemm2_0 from `/shared_nfs/zihao/headkernel_ut_0831` on 2026-09-14.
Prior optimization results (`_candidate_best/`, `accepted_overlay/`, tuning sweeps,
patches) were deliberately **not** copied - a benchmark that ships the answer
measures nothing. They remain in the upstream package.
Oracle blobs are hardlinked, not duplicated (1 file(s), 0.18 GB shared with the source package).
The original package README is preserved at `ut/README.md` and is the authority on
this op's measurement caveats - read it before trusting a speedup.

**Note.** same synthetic oracle layout as K3-4.
