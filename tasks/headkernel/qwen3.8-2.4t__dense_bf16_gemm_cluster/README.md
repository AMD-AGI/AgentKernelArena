# qwen3.8-2.4t__dense_bf16_gemm_cluster

**Qwen3.8-2.4T-A95B-MXFP4** head kernel - `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x240x64_MI -> MT256x256x64_MI` (hipBLASLt / Tensile, prefill).

| field | value |
|---|---|
| GPU time share | 15.56% |
| empirical roofline | 40.1% compute-bound |
| optimized roofline | 64.3% compute-bound |
| e2e uplift measured | E4 Prefill GEMM 5-row +1.563% paired / +2.306% native |
| device symbol | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x240x64_MI -> MT256x256x64_MI (ledger hk04 dispatches hipblaslt)` |
| production seam | `aiter.tuned_gemm:gemm_a16w16` |
| serving contract | ISL 8192 / OSL 1024 / CONC 64 / TP 8 |
| image | `harbor.crusoe.primus-safe.amd.com/hyperloom-image/sglang:v0.5.18-rocm720-mi35x-profilerfix` |
| owner | Li, Zeping <zepingl@amd.com> |
| info rows | Q38-4, Q38-5, Q38-6, Q38-7, Q38-8 |

## One seam, several device symbols

The info table lists these separately, but the profiler resolves all of them to
`aiter.tuned_gemm:gemm_a16w16`. They are one optimization target, not several:

- **Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x240x64_MI -> MT256x256x64_MI** - 15.56% GPU, roofline 40.1% compute-bound -> 64.3% compute-bound
- **gemm_a16w16 M64 N32 K8192 - torch solution:0 -> native (retained control)** - 21.43% GPU, roofline 1.95% memory-bound -> 1.95% memory-bound
- **gemm_a16w16 M64 N512 K8192 - torch solution:0 -> flydsl t16x64x256 split-K8** - 21.43% GPU, roofline 10.96% memory-bound -> 12.9% memory-bound
- **gemm_a16w16 M64 N4608 K8192 - torch solution:0 -> bf16gemm_fp32bf16_tn_64x64_splitk_clean** - 21.43% GPU, roofline 37.37% memory-bound -> 51.68% memory-bound
- **gemm_a16w16 M64 N8192 K256 - torch solution:0 -> flydsl t32x64x64 split-K1** - 21.43% GPU, roofline 9.23% memory-bound -> 13.98% memory-bound

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

- `gemm_a16w16` in `source/tuned_gemm_candidate.py`
- `torch_gemm` in `source/tuned_gemm_candidate.py`

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

Copied dense_bf16_gemm_task from `/shared_nfs/zepingl/qwen38/HeadKernel/20260913T180016Z/tasks` on 2026-09-14.
Prior optimization results (`_candidate_best/`, `accepted_overlay/`, tuning sweeps,
patches) were deliberately **not** copied - a benchmark that ships the answer
measures nothing. They remain in the upstream package.
This package has no oracle blob large enough to hardlink; everything was copied.
This package shipped no README. `ut/meta.json` (oracle policy, tolerance, case
geometries), `ut/selection_validation.json` (proof the two legs resolve to
different code) and `ut/negative_check.json` (proof a corrupted output is
rejected) are what it records instead - read those before trusting a speedup.

**Note.** Q38-4..Q38-8 are one aiter.tuned_gemm seam with five live shapes (ledger hk04-hk08): the M=16384 prefill GEMM plus the four M=64 decode shapes that share the 21.43% bucket. The oracle is a deterministic runtime torch.nn.functional.linear baseline, not a frozen tensor blob, so the performance leg uses the GEAK interleaved legs rather than record replay. gpu_pct 21.43 is a bucket SHARED by Q38-5..Q38-8 - one aiter.tuned_gemm seam, four decode shapes. It is not additive: do not sum it across those four rows. Identity caveat: this package's distinct_baseline_candidate gate compares the candidate against torch.nn.functional.linear rather than against the unshadowed aiter callable, and _callable_identity reports the decorator's source file, so the gate does not by itself prove the timed code is source/tuned_gemm_candidate.py. The overlay rebind IS observable in the run log ('[overlay] rebound aiter.tuned_gemm:gemm_a16w16 -> tuned_gemm_candidate.gemm_a16w16'); check for that line before trusting a speedup on this task.
