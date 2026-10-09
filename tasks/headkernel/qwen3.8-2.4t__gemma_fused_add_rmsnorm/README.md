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
scripts/_bench.py        canonical graph timing, 10 warmups / 100 raw samples
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
- **performance** builds the exact two live geometries through `ut/cases.py`
  and uses the workspace-materialized canonical `_aka_benchmark.py` helper.
  It performs 10 warmups and reports all 100 graph replay samples per case in
  their original order, together with per-case `benchmark_method: cuda_graph`.
  Each replay contains one native callable invocation. Before every replay,
  outside the event interval, it generates new BF16 x, residual and weight
  values on the CPU and poisons both output buffers with NaN. Timing inputs
  retain the original uniform domains: x `[-0.75, 0.875]`, residual
  `[-0.625, 0.5]`, and weight `[-0.125, 0.125]`. A continuous CPU RNG supplies
  fresh values before every warmup, capture, and graph replay. Expected outputs
  are computed entirely on the CPU from the unrounded FP32 residual sum and
  remain on the CPU; only inputs are copied to the GPU. Completed outputs are
  copied back and checked at the existing tolerance `0.02`, including the last
  timed sample and the exact graph whose samples were recorded. Validation,
  input generation/copies and output poisoning are excluded from device timing. Capture
  failure, invalid samples or either output failing validation fails the whole
  performance run; there is no event fallback for these graph-capable cases.
  After timing, that same graph is challenged with zero residuals, exactly
  cancelling residuals, near-cancelling residuals, and small finite amplitudes
  where `eps=1e-6` matters. These four correctness replays do not enter the 100
  reported samples. The correctness command also runs ordinary values and all
  four challenges at both live shapes after the original frozen unittest.
  A candidate that caches its first outputs is rejected when inputs change,
  including when its private cache survives poisoning of the public outputs.
  Use framework workspace setup or `make materialize-perf-task TASK=<task>`
  from the repository before running timing directly in a copied task.

Historical CUDA-event timings describe the previous benchmark method. This
harness change requires fresh trusted and framework GPU qualification; the old
numbers are not graph timings or a declared event fallback.

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

## SGLang 0.5.20 source refresh

The candidate and frozen native source now bind to the digest in `ut/runtime_refresh.json`. Existing shapes, captured tensors, tolerances, samples, and harness checks are preserved from the documented SGLang 0.5.18 workload. Historical PASS artifacts apply to that capture runtime. Validation and dispatch verification in SGLang 0.5.20 are pending; no new serving capture or performance gain is claimed.

The fused kernel computes the residual sum in FP32 and uses that unrounded sum for the variance and normalization. It casts independently when storing the BF16 normalized output and the BF16 residual output; the stored residual is not reread to compute the normalization. Both output buffers are fresh and distinct.
