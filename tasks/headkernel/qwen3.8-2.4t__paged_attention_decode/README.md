# qwen3.8-2.4t__paged_attention_decode

**Qwen3.8-2.4T-A95B-MXFP4** head kernel - `paged_attention_ll4mi_QKV_mfma16_kernel` (AITER / CK asm, decode).

| field | value |
|---|---|
| GPU time share | 6.84% |
| empirical roofline | 48.2% memory-bound |
| optimized roofline | - |
| e2e uplift measured | Stage-A scenario +0.25% (NOT measured) |
| device symbol | `paged_attention_ll4mi_QKV_mfma16_kernel` |
| production seam | `aiter.ops.attention:paged_attention_ragged` |
| serving contract | ISL 8192 / OSL 1024 / CONC 64 / TP 8 |
| image | `harbor.crusoe.primus-safe.amd.com/hyperloom-image/sglang:v0.5.18-rocm720-mi35x-profilerfix` |
| owner | Li, Zeping <zepingl@amd.com> |
| info rows | Q38-9 |

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

- `paged_attention_ragged` in `source/attention_candidate.py`

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
- `direct_register_custom_op`

## Starting point

`source/` is the **stock** file from the pinned runtime image plus 22 line(s)
of unit-test harness shim (an `importlib` import and a `baseline_callable()`
accessor, so the candidate overlay can still reach the unshadowed production
function). Nothing has been pre-optimized - the first measured speedup on this
task is by construction a null run, and small deviations from 1.00x are
timing-slot bias rather than optimization.

## Provenance

Copied paged_attention_decode_task from `/shared_nfs/zepingl/qwen38/HeadKernel/20260913T180016Z/tasks` on 2026-09-14.
Prior optimization results (`_candidate_best/`, `accepted_overlay/`, tuning sweeps,
patches) were deliberately **not** copied - a benchmark that ships the answer
measures nothing. They remain in the upstream package.
**1 oracle blob(s) (0.54 GB) are real copies, not hardlinks** - the upstream file belongs to another user and
`fs.protected_hardlinks` forbids linking it. Run `tools/relink_oracles.py` as root
to convert them back and reclaim the space.
This package shipped no README. `ut/meta.json` (oracle policy, tolerance, case
geometries), `ut/selection_validation.json` (proof the two legs resolve to
different code) and `ut/negative_check.json` (proof a corrupted output is
rejected) are what it records instead - read those before trusting a speedup.

**Note.** the live capture yielded exactly one decode geometry (M=64, 521351 referenced KV pages). The callable must return the supplied output buffer and leave query/KV/metadata/scales unchanged - the UT checks that, not just the values. source/ also stubs direct_register_custom_op so loading the overlay copy does not re-register the torch.library schemas the production module already owns; keep it.
