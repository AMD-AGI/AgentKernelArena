# qwen3.8-2.4t__gemma_fused_add_rmsnorm

**Qwen3.8-2.4T-A95B-MXFP4** head kernel - `_gemma_fused_add_rmsnorm_kernel` (Triton, prefill+decode).

| field | value |
|---|---|
| GPU time share | 5.06% |
| empirical roofline | 77.9% memory-bound |
| optimized roofline | - |
| e2e uplift measured | planning scenario ~+0.24% (NOT measured) |
| device symbol | `_gemma_fused_add_rmsnorm_kernel` |
| production seam | `sglang.srt.layers.layernorm:rocm_triton_gemma_fused_add_rmsnorm` |
| serving contract | ISL 8192 / OSL 1024 / CONC 64 / TP 8 |
| image | `harbor.crusoe.primus-safe.amd.com/hyperloom-image/sglang:v0.5.18-rocm720-mi35x-profilerfix` |
| owner | Li, Zeping <zepingl@amd.com> |
| info rows | Q38-11 |

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

- `gemma_fused_add_rmsnorm` in `source/minimax_m3_rmsnorm.py`
- `_gemma_fused_add_rmsnorm_kernel` in `source/minimax_m3_rmsnorm.py`

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
- **correctness** runs `ut/unittest.py`. This op is value-independent, so instead
  of freezing hundreds of MB of tensors the UT regenerates a deterministic
  baseline at run time at tol `0.02`, over the live shapes, strides
  and dispatch selections recorded in `ut/meta.json`. Its own gates are a
  deliberate output corruption that must be rejected (`ut/negative_check.json`)
  and an identity check that the two legs resolve to different code
  (`ut/selection_validation.json`).
- **performance** builds this op's live geometries through the package's own
  `ut/cases.py` - there is no frozen blob to replay - and times them with the same
  10 warmup + 100 measured cuda-event methodology. It falls back to the GEAK
  interleaved median-of-3 legs only if that entry point is missing, and says which
  it used in `build/performance_report.json`.
  A run whose unit test did not pass reports no cases at all.

## Starting point

`source/` is the **stock** upstream code, byte-identical to
`ut/baseline_ref/*.orig`. Nothing has been pre-optimized.

## Provenance

Copied fused_add_rmsnorm_task from `/shared_nfs/zepingl/qwen38/HeadKernel/20260913T180016Z/tasks` on 2026-09-14.
Prior optimization results (`_candidate_best/`, `accepted_overlay/`, tuning sweeps,
patches) were deliberately **not** copied - a benchmark that ships the answer
measures nothing. They remain in the upstream package.
This package has no oracle blob large enough to hardlink; everything was copied.
This package shipped no README. `ut/meta.json` (oracle policy, tolerance, case
geometries), `ut/selection_validation.json` (proof the two legs resolve to
different code) and `ut/negative_check.json` (proof a corrupted output is
rejected) are what it records instead - read those before trusting a speedup.

**Note.** prefill M=8192 and decode M=64 at N=8192. The oracle is a deterministic runtime baseline rather than a frozen blob; source/ is byte-identical to the deployed kernel, so the first measurement is a null run.
