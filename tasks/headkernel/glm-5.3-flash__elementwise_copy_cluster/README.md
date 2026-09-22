# glm-5.3-flash__elementwise_copy_cluster

**GLM-5.3-Flash** head kernel - `elementwise_kernel_manual_unroll` (ATen direct_copy_kernel_cuda, decode).

| field | value |
|---|---|
| GPU time share | 3.24% |
| empirical roofline | 0.0006 |
| optimized roofline | 0.0009 |
| e2e uplift measured | - |
| device symbol | `elementwise_kernel_manual_unroll<128,4,...direct_copy_kernel_cuda...>` |
| production seam | `sglang.srt.layers.quantization.fp8_utils:materialize_bpreshuffle_fp8_scale` |
| serving contract | ISL 8192 / OSL 1024 / CONC 64 / TP 8 |
| image | `harbor.crusoe.primus-safe.amd.com/hyperloom-image/sglang:v0.5.17-rocm720-mi35x-profilerfix` |
| owner | Hongtaom |
| info rows | G53-5 |

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

- `materialize_bpreshuffle_fp8_scale` in `source/bpreshuffle_scale_impl.py`

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
  at tol `1e-06`. Exit 0 pass, 1 correctness fail, 2 environment,
  3 harness incomplete.
- **performance** replays the captured argument records from `ut/reference_io.pt`
  with 10 warmup + 100 measured iterations and reports the mean cuda-event device
  time. If a record cannot be rebuilt it falls back to the GEAK interleaved
  median-of-3 legs and says so in `build/performance_report.json`.
  A run whose unit test did not pass reports no cases at all.

## Starting point

This op is optimized by **rebinding the seam to a new module**, so the file in
`source/` has no upstream counterpart - it is itself a candidate implementation
rather than stock library code. Speedup is still measured against the live
stack (the baseline leg resolves outside the task dir), but be aware the
starting point already encodes design choices from the capture.

## Provenance

Copied GLM-5.3-Flash_elementwise_copy_cluster_0913 from `/shared_nfs/hongtaom/headkernel_ut_0913` on 2026-09-14.
Prior optimization results (`_candidate_best/`, `accepted_overlay/`, tuning sweeps,
patches) were deliberately **not** copied - a benchmark that ships the answer
measures nothing. They remain in the upstream package.
This package has no oracle blob large enough to hardlink; everything was copied.
The original package README is preserved at `ut/README.md` and is the authority on
this op's measurement caveats - read it before trusting a speedup.

**Note.** 1.4253x isolated, e2e never measured. Correctness is two-part: values AND transposed-contiguous physical stride.
