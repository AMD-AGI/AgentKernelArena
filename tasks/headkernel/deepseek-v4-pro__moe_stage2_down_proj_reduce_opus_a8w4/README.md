# deepseek-v4-pro__moe_stage2_down_proj_reduce_opus_a8w4

**DeepSeek-V4-Pro** head kernel - `MoE stage-2 opus_moe_stage2_a8w4_decode_kernel_gfx950` (aiter, decode+prefill).

| field | value |
|---|---|
| GPU time share | 3.7% |
| empirical roofline | ~98% HBM (raw 144%, byte model over-counts) |
| optimized roofline | - |
| e2e uplift measured | - |
| device symbol | `opus_moe_stage2_a8w4_decode_kernel_gfx950` |
| production seam | `aiter.ops.opus.moe_stage2_a8w4_fused_adapter:opus_a8w4_stage2_wrapper` |
| serving contract | ISL 8192 / OSL 1024 / CONC 64 / TP 8 |
| image | `harbor.crusoe.primus-safe.amd.com/hyperloom-image/sglang:v0.5.17-rocm720-mi35x-profilerfix` |
| owner | zihao |
| info rows | DS-3 |

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

- `opus_a8w4_stage2_wrapper` in `source/geak_opus_moe_stage2_adapter.py`

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

`source/` is seeded from the **stock** pre-optimization code
(`baseline_ref/*.orig` in the upstream package). The package itself shipped a
kernel_src that a previous GEAK run had already tuned (1 file(s)); that version is kept out of the task at
`_prior_solutions/deepseek-v4-pro__moe_stage2_down_proj_reduce_opus_a8w4/` so this benchmark starts where every other task in
the suite starts.

## Provenance

Copied DeepSeek-V4-Pro_moe_stage2_down_proj_reduce_opus_a8w4 from `/shared_nfs/zihao/headkernel_ut_0831` on 2026-09-14.
Prior optimization results (`_candidate_best/`, `accepted_overlay/`, tuning sweeps,
patches) were deliberately **not** copied - a benchmark that ships the answer
measures nothing. They remain in the upstream package.
**1 oracle blob(s) (2.22 GB) are real copies, not hardlinks** - the upstream file belongs to another user and
`fs.protected_hardlinks` forbids linking it. Run `tools/relink_oracles.py` as root
to convert them back and reclaim the space.
The original package README is preserved at `ut/README.md` and is the authority on
this op's measurement caveats - read it before trusting a speedup.
