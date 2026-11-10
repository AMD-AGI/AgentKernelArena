# bf16 GEMM, N=6288, K=7168

Dense bfloat16 matrix multiplication in the NT layout, `out = a @ b.T`, with `a[M, 7168]` activations,
a `[6288, 7168]` row-major weight `b` and a `[M, 6288]` bfloat16 output. Accumulation is in float32 or
better and the result is rounded to bfloat16 once. `M` is the only varying axis. The operator, input
names, dtypes and distributions are those of the other `gemm_a16w16` scenarios; this scenario has its
own production origin and baseline.

## Production origin

The fused KDA input projection (`in_proj`) of Kimi-K3.1's Kimi Delta Attention layers at tensor
parallel 8: q, k and v (3 x 1536 rows), the output gate (1536), the forget-gate low-rank input (128) and
beta (12) of a rank's 12 heads, 6284 rows padded to 6288, against the 7168-wide hidden state. In
sgl-amd-mono (`sgl-project/sgl-amd-mono` at `52486703`) this GEMM runs inside one persistent launch per
KDA layer for wide verify steps (`kimi_k31/gfx950/layer.py`, stage T3), on the row buckets of
`kimi_k31/wide.py`. Steps of up to 8 rows run Kimi-K3's narrow launch instead, which fuses the same
projection into its KDA front.

## Baseline

`baseline/tile_mma.py` is sgl-amd-mono's LDS tile GEMM stage (`kimi_k31/gfx950/stages/tile_mma.py`):
a CTA of 8 waves computes a `bm x bn` output tile over one K slice, streaming x and W tiles three
64-wide K steps ahead through double-buffered, padded LDS into `mfma_f32_16x16x32_bf16` (gfx950 only).
The device function `mma_tile` is the one the layer launch runs; the task launches it through the
stage's `standalone` build, as `test/kernels/kimi_k31/mma_bench.py` does.

`baseline/kernel.py` vendors the layer's tuned configuration table (`CFGS`, `INP_ROWS`, `cfg_rows`) and
selects as production does: M rows take the smallest 8-token (DSpark, 7 draft tokens + 1) bucket of
16, 32, 64, 128, 256 or 384 rows that holds them, and that bucket's in_proj entry
(`16:128:1:8:4` at 16 rows, `128:128:2:4:4` at 128, `192:128:4:2:2` at 384, as `bm:bn:wm:wn:ks`).
Rows are padded to the bucket as the layer launch pads them (a same-dtype copy of `a`; padded rows are
computed and never read). Every in_proj entry splits K across `ks` CTAs. The layer merges the slices
with packed-bf16 atomic adds and finishes the tile inside the launch; standalone, the stage writes f32
partials (`[ks, bucket, 6288]`) and a small FlyDSL kernel in `kernel.py` sums them in slice order and
rounds to bfloat16 once. The baseline is therefore two launches and is bitwise deterministic. M up to
8, which production serves with Kimi-K3's narrow launch, takes the 16-row bucket.

`stages/tile.py` (`k31.tile`) is an alternative implementation of the same operator: operands go
straight to registers with nontemporal weight loads, K is split across the 8 waves of a CTA and reduced
through LDS, and the output is written in one launch. It is not the baseline: the layer launch and its
tuned table use `mma_tile`, and `tile.py`'s register footprint grows with M (one x operand per 16 rows
per in-flight unit). Only its `bf16_emit` and `f32_emit` epilogues are vendored (`baseline/tile.py`).

FlyDSL 0.3.2: the vendored code uses no API newer than the runtime's (no `fx.generic_load` /
`generic_store` / `memory_fence` / `AtomicOrdering`), so sgl-amd-mono's `muse_spark/gfx950/compat.py`
bridge is not needed. `baseline/buffer_ops.py` is ROCm/aiter's copy of FlyDSL's buffer operations with
the 0.3.2 cache-policy adaptation, the same file the `paged_mqa_logits` task bundles. Modifications
are listed in `task.yaml`. Licenses: `baseline/LICENSE` (sgl-amd-mono Apache-2.0 and the ROCm/ATOM MIT
notice for `ops.py`) and `baseline/LICENSE-FlyDSL`.

## Workloads

Monokernel steps are decode and speculative verify only; prefill shapes do not apply, and the axis
ends at 384 rows, the largest wide bucket. Rows come from the serving configuration (Kimi-K3.1 at TP8
with DSpark, 8 rows a request): narrow steps of 1 to 8 rows and wide verify steps of 2 to 48 requests.
A GEMM has no context axis, so the AgentX corpus statistics do not select the rows; the cases follow
the coordinator's workload guide, derived from the corpus and the serving configuration, and are not
yet confirmed by a live capture. Request-count frequencies are not known, so no case is weighted.

Performance cases M = 1, 8, 16, 96, 128, 288, 384 (a 384x span) use three buckets (16, 128 and 384
rows), so an evaluation builds three GEMM configurations and three partial-sum kernels per
implementation. Every case reads the 90 MB weight once (the 384-row bucket's second row tile reads it
again), so even M=1 needs about 11 us at the MI355X peak HBM bandwidth after the cold-cache flush.
Correctness-only cases add M=3 and the unaligned M=100 and M=333 (the latter ends inside the second
192-row tile). Inputs are standard-normal for both operands, as in the other scenarios.

Estimated baseline times (not measured; from bytes moved at about 5 TB/s and the MFMA rate): about
20-26 us for M <= 16, 25-33 us for M = 96 and 128, and 40-50 us for M = 288 and 384.

## Correctness bound

`compare.out` is the default `baseline_relative` bound with `k` = 2. The other `gemm_a16w16` scenarios
use `k` = 8 because their split-K baselines add partial sums with atomics and are not deterministic;
this baseline sums its K slices in a fixed order in float32 and rounds once, so its error is one
bfloat16 rounding and an exact copy reproduces it bitwise. The operator rounds once, so a candidate
must accumulate in float32 or better; any order passes. A candidate that merges K slices with
bfloat16 atomic adds, as the layer launch does, rounds once a slice and fails.

A CPU emulation of the baseline's arithmetic (f32 slice partials, ordered sum, one rounding) against
the float64 reference with the evaluator's comparison, standard-normal inputs, N=6288: a one-pass
float32 GEMM passes at 0.33x of the bound (M=8, 100, 128, 288, 384); merging the slices with bfloat16
adds peaks at 1.7x to 3.1x and fails; dropping a K slice (460x and more), fp8 rounding of `a` (29x and
more) and a 16-ulp output shift (25x) fail far outside it. At `k` = 8 the bfloat16 merge would peak at
0.6x to 0.99x, so even the contract maximum would not admit it reliably.

## Risks for the GPU audit

Nothing here has run on a GPU: the kernel, the partial-sum kernel, the timings above and the bound
calibration are unverified. The vendored `mma_tile` and dispatch table are what sgl-amd-mono runs on
MI355X; tracing it on the runtime's FlyDSL 0.3.2 (checked against the 0.3.2 sources by inspection only),
the standalone build at these exact (rows, configuration) pairs and the partial-sum kernel (new code)
are the parts most likely to fail first. The FlyDSL compile time of the 384-row configuration (56
unrolled K steps of 24 MFMAs a wave) sets most of the evaluation time; three builds per
implementation are expected to fit the 10-minute budget.
