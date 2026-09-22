# glm-5.3-flash__fused_moe_kernel

**GLM-5.3-Flash** head kernel - `fused_moe_kernel` (Triton - sglang srt/layers/moe/moe_runner/triton_utils.fused_moe, decode).

| field | value |
|---|---|
| GPU time share | 23.32% |
| empirical roofline | 0.980 |
| optimized roofline | - |
| e2e uplift measured | 3.48% |
| device symbol | `fused_moe_kernel` |
| production seam | `sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe:fused_experts_impl` |
| serving contract | ISL 8192 / OSL 1024 / CONC 64 / TP 8 |
| image | `harbor.crusoe.primus-safe.amd.com/hyperloom-image/sglang:v0.5.18-rocm720-mi35x-profilerfix` |
| owner | Hongtaom |
| info rows | G53-1 |

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

- `fused_experts_impl` in `source/fused_moe.py`
- `fused_experts` in `source/fused_moe.py`

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

Copied GLM-5.3-Flash_fused_moe_kernel from `/shared_nfs/hongtaom/headkernel_ut_0913` on 2026-09-14.
Prior optimization results (`_candidate_best/`, `accepted_overlay/`, tuning sweeps,
patches) were deliberately **not** copied - a benchmark that ships the answer
measures nothing. They remain in the upstream package.
Oracle blobs are hardlinked, not duplicated (1 file(s), 1.04 GB shared with the source package).
The original package README is preserved at `ut/README.md` and is the authority on
this op's measurement caveats - read it before trusting a speedup.

**Note.** TWO blockers, in firing order. (1) IMAGE: runs only on sglang:v0.5.18-rocm720-mi35x-profilerfix - the stack it was captured on per its own ut/README.md. Under the model-level v0.5.17 pin it dies first at ut/sglang_bootstrap.py:39 with AttributeError: 'RuntimeContext' object has no attribute 'is_config_namespace_published', which is a 0.5.18+ API. This row now carries a docker override. (2) ARCH PATCH: with the right image but no patch it dies next at sglang_bootstrap.py:43 with ValueError: model type 'glm5_next' not recognized - it is the only package that boots sglang's ServerArgs against the real checkpoint, and the stock image registers no glm5_next architecture. Apply /shared_nfs/hongtaom/qwen3_14B/hl_matrix_0824/patches/glm53-flash/001_glm5_next_arch_enablement.patch to /sgl-workspace/sglang before running. Both were verified green together on 2026-09-13 by the package owner. NOTE also that an uncaught harness exception here surfaces as exit=1, which the runner labels 'correctness FAIL' - it is an environment failure; ut/unittest.py:47 lacks the try/except its neighbours have.
