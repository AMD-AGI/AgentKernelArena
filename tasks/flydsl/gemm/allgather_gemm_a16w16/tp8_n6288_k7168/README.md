# All-gather + bf16 GEMM, TP8, N=6288 a rank, K=7168

One decode or verify step's all-gather of the activations fused with the dense bfloat16 GEMM that consumes
them, on the 8 ranks of a tensor-parallel group. The hidden state `x_full [m, 7168]` is column-sharded:
rank `r` holds `x = x_full[:, 896 r : 896 (r + 1)]`. Every rank also holds its own weight
`w [6288, 7168]` (row-major, out x in) and writes

```
out[m, 6288] = x_full @ w^T        (bfloat16 in, float32 or better accumulation, bfloat16 out)
```

so a rank needs every other rank's slice before its K range can be reduced. The task is one call per
rank; the ranks exchange the slices inside the kernel through the framework's symmetric (peer) memory
(`peers`, `symmetric`; see [multi-GPU tasks](../../../../../docs/how-to/add-task.md#multi-gpu-tasks)).
`m` (1 to 384) is the only varying axis.

## Production origin

Kimi-K3.1 at TP8: the fused KDA input projection (`in_proj`: q, k and v of 3 x 1536 rows, the output
gate, the forget-gate low-rank input and beta of a rank's 12 heads, 6284 rows padded to 6288) reads the
all-gathered output of the attention AttnRes seam, whose residual stream is column-sharded (896 columns a
rank). sgl-amd-mono (`sgl-project/sgl-amd-mono` at `52486703`) fuses this all-gather and in_proj into
its wide-row KDA layer launch (`kimi_k31/gfx950/layer.py`, steps of 16 to 384 rows), and Kimi-K3's narrow
launch serves steps up to 8 rows with the same projection. `kimi_k31/gfx950/stages/gemm_coll.py::ag_gemm`
is that all-gather + GEMM as one standalone launch, driven by `test/kernels/kimi_k31/ag_gemm_bench.py` at
exactly this shape (its defaults: n 6288, k 7168): benchmark-only upstream, but the operator of the
layer's in_proj stage. The task's GEMM shape is that of the single-GPU sibling
`tasks/flydsl/gemm/gemm_a16w16/n6288_k7168`, which has no all-gather.

## Baseline

`baseline/gemm_coll.py::ag_gemm` builds one persistent launch of 256 CTAs x 512 threads per rank:

1. **Push.** CTA group `j` (32 CTAs) copies this rank's `x` into rank `j`'s ag region (`[rows, 7168]`
   bf16, at columns `896 rank ..`) with plain 16-byte system-scope stores over XGMI, its own rank
   included. Each CTA drains its stores (`s_waitcnt vmcnt(0)`) and counts itself in a local arrival
   counter (an agent-scope atomic add); the 256th arrival writes a tagged flag `(rank, epoch tag)` into
   every rank's flag region.
2. **GEMM.** Tasks are `[bm, bn]` output tiles over one of `ks` K slices (`tile_mma.mma_tile`: x and W
   tiles three 64-wide K steps ahead through double-buffered LDS into `mfma_f32_16x16x32_bf16`, gfx950
   only). A task first polls the flags of the ranks that own its K range (2 owners at `ks` = 4, 4 at
   `ks` = 2), then reads x from this rank's ag region. Weights stream with nontemporal loads.
3. **Merge ("linbf").** The epilogue shuffles the f32 tile through LDS and adds it to the zeroed bf16
   `out` with lane-contiguous packed-bf16 atomic adds (`tile.atomic_lin`): each K slice is rounded to
   bf16 and the slices merge in arrival order.

`baseline/kernel.py::run` takes the bench's default launch configuration (`--cfgs`, the in_proj
entries of `kimi_k31/wide.py`'s tuned table at the same rows) of the smallest of its 16-, 32-, 128- and
384-row builds that holds `m`:

| build rows | m | bm:bn:wm:wn:ks |
| --- | --- | --- |
| 16 | 1-16 | 16:128:1:8:4 |
| 32 | 17-32 | 32:128:2:4:4 |
| 128 | 33-128 | 128:128:2:4:4 |
| 384 | 129-384 | 192:128:4:2:2 |

zeroes `out` (the bench includes this memset in every timed launch), and passes the bench's launch
arguments: epoch word 0 = 1 (its first epoch bump) and slot 0 of a one-launch step. The peer buffer
(`coll.peer_bytes(rows, 7168, "flag")`: two parity halves of rs / ag / flag regions, then the guard's
error flags, 22.1 MB at 384 rows) is the start of `symmetric`; the arrival counter follows it. The
framework zeroes `symmetric` before every call, in place of the bench's zeroed-once buffers plus its
per-launch epoch bump, so a call's tags never meet an earlier call's words. `epoch` is the guard block:
the epoch, a 60 s wait deadline (a wait that times out sets an error word, flags every peer and stops the
launch) and clean error words.

**One kernel change.** Upstream `ag_gemm` compiles its row count `m` into the push loop, the task count,
the x-load clamp and the output masks. Here the build's `m` is the row bucket (it still sets the peer
layout and the configuration) and the launch takes its row count as a kernel argument, so one build
serves every row count of its bucket and a launch with `rows == m` does exactly the upstream build's work
(for m = 136 in the 384-row build, one 192-row tile instead of two). This keeps the evaluation at four
compiled builds an implementation. Other changes are vendoring only (listed in `task.yaml`): flattened
imports, the definitions this launch uses, the FlyDSL 0.3.2 forms of the launch stop and the arrival
counter's atomic (`compat.py`; the runtime has no `fx.llvm`), and only the production-relevant "linbf"
merge (no "part" f32 partials, ks = 1 store, timing-only phases or stress delays). The common modules
are the `muse_spark_layer` task's FlyDSL 0.3.2 copies; `buffer_ops.py` is ROCm/aiter
`dbd8bf5bd`'s (cache policy as an op attribute, no environment read). Licenses:
`baseline/LICENSE` (sgl-amd-mono Apache-2.0 with its NOTICE and the ROCm/ATOM MIT notice, FlyDSL's
`buffer_ops.py` Apache-2.0).

## Inputs and shards

`inputs.py::make_inputs` draws the logical inputs once per case and seed, as the bench does:
`x_full ~ N(0, 1)` and the in_proj weight of all 8 ranks `w ~ N(0, 1) / sqrt(7168)` (`[8 x 6288, 7168]`,
rank `r`'s rows at `6288 r ..`), both rounded to bf16, so `out` is about N(0, 1). `shard_inputs` gives
rank `r` the columns `896 r ..` of `x_full` and its rows of `w`; `rank_expected` gives it the columns
`6288 r ..` of the logical output. `reference.py` computes `x_full @ w^T` in float64 from the bf16 inputs,
one rank's rows at a time.

## Workloads

Monokernel steps are decode and speculative verify only; prefill shapes do not apply. Rows come from the
serving configuration (Kimi-K3.1 at TP8 with DSpark, 8 rows a request): narrow steps of 1 to 8 rows and
wide verify steps of 2 to 48 requests (16 to 384 rows). A GEMM has no context axis. Row frequencies are
derived from the AgentX corpus (`semianalysisai/cc-traces-weka-062126`, the per-session in-flight request
distribution: k = 1 90.0%, 2 5.4%, 3 2.1%, ...) at concurrency points C in {1, 2, 4, 8, 16, 32, 48},
equally weighted, as the distribution of the sum of C sessions' in-flight requests, restricted to steps of
up to 48 requests and renormalized. Each 8-row step band holds about 13-16% (8 rows 15.1%, 16 rows 14.5%,
24-32 13.8%, 40-64 13.2%, 72-128 13.2%, 136-256 14.5%, 264-384 15.7%), so the performance cases take one
high-frequency row count from each band plus the axis ends: m = 1, 8, 16, 32, 64, 128, 136, 288, 384
(nine cases, a 384x span, so no span exemption is needed). m = 1 (a step without speculation) is the
axis minimum; DSpark serving gives it no weight. These shapes are derived from corpus statistics and the
serving configuration and not yet confirmed by a live capture. Correctness-only cases add m = 3
(non-power-of-two), 27 (not a multiple of 16, > 16, in the 32-row build), 100 (unaligned in the 128-row
tile) and 333 (ending inside the second 192-row tile).

Estimated baseline times (not measured; from bytes moved at about 5 TB/s and the MFMA rate, with a few
microseconds for the push and flag round trip and the `out` memset): every case reads each rank's 90 MB
weight (18 us at 5 TB/s), so about 23-32 us for m <= 32, 28-38 us for 64 to 136 (one row tile, 200 or 100
GEMM tasks) and 45-58 us for 288 and 384 (two 192-row tiles: the weight is read twice, partly from the
MALL). Every performance case is above the 10 us floor with margin.

Evaluation budget: four builds an implementation (16, 32, 128 and 384 rows; the 384-row build unrolls 56
K steps of 24 MFMAs a wave, like the single-GPU sibling's, whose whole QA evaluation took 33 s);
correctness runs 13 cases x 2 seeds and the timing 9 cases x 110 replays a rank (about 3-4 ms a sample
with the flush and the synchronized release). The estimate for the whole QA run, dominated by starting 8
rank processes per action and by the compiles, is 4-7 minutes (not measured).

## Comparison

`compare.out` is `sqnr` with `min_db` 46. Compared with the default (`baseline_relative`, k 2) this **changes**
the criterion from element-wise to aggregate: it **loosens** the check for a few wrong elements (a whole-output
SQNR does not see isolated ones; one missing row of 384 is still rejected) and does not depend on the run.
`baseline_relative` does not fit: the linbf merge rounds each K
slice to bf16 and adds the slices with bf16 atomics in arrival order, so for the `ks` = 4 builds (m <= 128)
`out` is not bitwise deterministic run to run, and a correct candidate that accumulates in float32 and
rounds once is about 30 dB more accurate than the baseline, so the baseline's own error is no measure of
an acceptable one. With `ks` = 2 the two-slice merge is commutative and the 384-row build is
deterministic.

A CPU emulation of the baseline's arithmetic (f32 slice partials, each rounded to bf16, merged by bf16 adds
in a random order; standard-normal inputs as above, 1024 output columns) measures, against the bf16-rounded
float64 reference that the `sqnr` bound uses: about 49 dB at `ks` = 4 and 51 dB at `ks` = 2 (50 and 52 dB
against the unrounded float64 values), with 44% of the elements differing between two arrival orders at
`ks` = 4. A single float32 accumulation rounded once measures 82-99 dB. Known-wrong outputs: one K slice
missing about 6 dB, one rank's x slice stale (another draw) 6 dB or zero 9 dB, a missing 128-column tile
17 dB (of 6288 columns), one missing row of 384 about 26 dB, the auditor's 16-ulp output shift 24 dB and
fp8 rounding of the output or of x about 31.5 dB.

Measured on MI355X: the baseline re-run 4 times eagerly and 40 times as cold-cache graph replays per case,
both seeds, all 13 cases on all 8 ranks, on two nodes (9984 comparisons): the default bound rejected most eager
re-runs of the baseline itself (1073 of 1664), and the lowest SQNR was 49.7 dB (node spread 0.1 dB). `min_db`
is that value rounded down, minus 3 dB. Single-output mutants of a baseline run, judged by the evaluator's own
comparison in every case on every rank (104 each), were all rejected at the provisional 40 dB already: the
first eighth of the last dimension zeroed, a 1% relative shift and Gaussian noise of 1% of the RMS; only a 0.1%
shift (below one bf16 step) passed. A float32 evaluation of the reference passes far above the bound.

## Risks for the GPU audit

Written before the GPU runs; the task has since passed the QA auditor on MI355X and the bound was calibrated as above. The parts most
likely to fail first:

- the runtime row count: tracing `ag_gemm` with a traced task-loop bound, push-loop bound and output-mask
  size on FlyDSL 0.3.2 (dynamic `range` bounds are used by other FlyDSL 0.3.2 baselines);
- `compat.global_atomic_add` (the same llvm `atomicrmw` as the Muse Spark experts task's counter) on the
  arrival counter in uncached symmetric memory;
- the push / flag protocol under the framework's process model (per-call zeroing, the parity half of
  epoch 1, the 60 s wait deadline against compile-time skew between ranks on a call's first launch);
- whether gfx950's packed-bf16 atomic add rounds to nearest even as the emulation assumes (a truncating
  add would lower the baseline's SQNR by a few dB).
