# qwen3.8-2.4t__fused_moe_2stage_mxfp4

**Qwen3.8-2.4T-A95B-MXFP4** head kernel - `mfma_moe1_silu_mul_afp4_wfp4_bf16_t32x128x256_pm1_async_v32 -> t32x32x256` (AITER / FlyDSL, decode).

| field | value |
|---|---|
| GPU time share | 21.06% |
| empirical roofline | 73.05% memory-bound |
| optimized roofline | 82.37% memory-bound |
| e2e uplift measured | E1 FMoE 6-row +1.025%; E3 E1+E2 native stack +3.922% |
| device symbol | `mfma_moe1_silu_mul_afp4_wfp4_bf16_t32x128x256_pm1_async_v32` |
| production seam | `aiter.fused_moe:fused_moe` |
| serving contract | ISL 8192 / OSL 1024 / CONC 64 / TP 8 |
| image | `harbor.crusoe.primus-safe.amd.com/hyperloom-image/sglang:v0.5.18-rocm720-mi35x-profilerfix` |
| owner | Li, Zeping <zepingl@amd.com> |
| info rows | Q38-1, Q38-2, Q38-3 |

## One seam, several device symbols

The info table lists these separately, but the profiler resolves all of them to
`aiter.fused_moe:fused_moe`. They are one optimization target, not several:

- **mfma_moe1_silu_mul_afp4_wfp4_bf16_t32x128x256_pm1_async_v32 -> t32x32x256** - 21.06% GPU, roofline 73.05% memory-bound -> 82.37% memory-bound
- **mfma_moe2_afp4_wfp4_bf16_cshuffle_t32x128x256 [atomic_bnt2 -> atomic]** - 11.58% GPU, roofline 75.47% memory-bound -> 71.65% memory-bound
- **FMoE 2-stage - S1 t64x128 + S2 atomic -> S1 t128x64 + S2 atomic_persist** - 15.73% GPU, roofline 22.6% memory-bound -> 23.7% memory-bound

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

- `fused_moe` in `source/moe_candidate.py`

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

Must survive in `source/` (not optimization targets - the two-leg UT resolves its
frozen baseline through them, and `task_runner.py compile` fails if one is gone):

- `baseline_callable`
- `_BASELINE`

## Starting point

`source/` is the **stock** file from the pinned runtime image plus 10 line(s)
of unit-test harness shim (an `importlib` import and a `baseline_callable()`
accessor, so the candidate overlay can still reach the unshadowed production
function). Nothing has been pre-optimized - the first measured speedup on this
task is by construction a null run, and small deviations from 1.00x are
timing-slot bias rather than optimization.

## Provenance

Copied fused_moe_mxfp4_task from `/shared_nfs/zepingl/qwen38/HeadKernel/20260913T180016Z/tasks` on 2026-09-14.
Prior optimization results (`_candidate_best/`, `accepted_overlay/`, tuning sweeps,
patches) were deliberately **not** copied - a benchmark that ships the answer
measures nothing. They remain in the upstream package.
**1 oracle blob(s) (2.08 GB) are real copies, not hardlinks** - the upstream file belongs to another user and
`fs.protected_hardlinks` forbids linking it. Run `tools/relink_oracles.py` as root
to convert them back and reclaim the space.
This package shipped no README. `ut/meta.json` (oracle policy, tolerance, case
geometries), `ut/selection_validation.json` (proof the two legs resolve to
different code) and `ut/negative_check.json` (proof a corrupted output is
rejected) are what it records instead - read those before trusting a speedup.
