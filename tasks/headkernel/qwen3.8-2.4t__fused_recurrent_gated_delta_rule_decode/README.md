# qwen3.8-2.4t__fused_recurrent_gated_delta_rule_decode

**Qwen3.8-2.4T-A95B-MXFP4** head kernel - `fused_recurrent_gated_delta_rule_packed_decode_kernel` (Triton, decode).

| field | value |
|---|---|
| GPU time share | 6.44% |
| empirical roofline | 53.5% memory-bound |
| optimized roofline | - |
| e2e uplift measured | Stage-A scenario +2.39% (NOT measured) |
| device symbol | `fused_recurrent_gated_delta_rule_packed_decode_kernel` |
| production seam | `sglang.srt.layers.attention.linear.kernels.gdn_triton:fused_recurrent_gated_delta_rule_packed_decode` |
| serving contract | ISL 8192 / OSL 1024 / CONC 64 / TP 8 |
| image | `harbor.crusoe.primus-safe.amd.com/hyperloom-image/sglang:v0.5.18-rocm720-mi35x-profilerfix` |
| owner | Li, Zeping <zepingl@amd.com> |
| info rows | Q38-10 |

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

- `fused_recurrent_gated_delta_rule_packed_decode` in `source/fused_recurrent.py`
- `fused_recurrent_gated_delta_rule_packed_decode_kernel` in `source/fused_recurrent.py`

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

Copied gated_delta_decode_task from `/shared_nfs/zepingl/qwen38/HeadKernel/20260913T180016Z/tasks` on 2026-09-14.
Prior optimization results (`_candidate_best/`, `accepted_overlay/`, tuning sweeps,
patches) were deliberately **not** copied - a benchmark that ships the answer
measures nothing. They remain in the upstream package.
**1 oracle blob(s) (1.04 GB) are real copies, not hardlinks** - the upstream file belongs to another user and
`fs.protected_hardlinks` forbids linking it. Run `tools/relink_oracles.py` as root
to convert them back and reclaim the space.
This package shipped no README. `ut/meta.json` (oracle policy, tolerance, case
geometries), `ut/selection_validation.json` (proof the two legs resolve to
different code) and `ut/negative_check.json` (proof a corrupted output is
rejected) are what it records instead - read those before trusting a speedup.

**Note.** two live records, B=64 and B=1, compared as an ordered pair: the UT checks the recurrent state transition, not just the output tensor.
