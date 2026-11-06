# LayerNorm, bfloat16, 256 features

Affine LayerNorm over small rows, as used between the layers of Meta's HSTU generative recommenders:
`out = (x - mean) * rstd * weight + bias` with `rstd = 1 / sqrt(var + epsilon)` and `epsilon = 1e-6`. Besides `out`
(bfloat16) the forward writes the per-row `mean` and `rstd` in float32, which the backward pass reuses; all three
are checked against a float64 reference with the baseline-relative bound (see Comparison below). `weight` and `bias` are bfloat16,
drawn uniform in [-1, 1) (the original task's distribution).

## Baseline

`_weighted_layer_norm_fwd` from `generative_recommenders/ops/triton/triton_layer_norm.py` at commit
`4f1e433ae1b241df074403c2da0c314a949332dd`: a multi-row Triton kernel (`BLOCK_N` rows per program, one block
pointer over the 256 features) chosen by `triton.autotune`. The kernel is unchanged. The unweighted kernel, the
backward kernels and the swish variant are removed, and the AMD autotune configuration list is inlined. One change
to the dispatch: the autotune key is `[BLOCK_D, N]` instead of `[BLOCK_D]` so that every row count is tuned on its
own shape and the baseline timing does not depend on which case ran first. Tuning happens during the warmup calls.

## Comparison

`out` uses the default baseline-relative bound. `mean` uses the baseline-relative bound with the default `k` = 2
and `max_violation_fraction` = 0.01; `rstd` uses a relative tolerance (below). Both are float32 reductions over the 256 features whose rounding depends
on the reduction tree, and the baseline's tree depends on the configuration `triton.autotune` picks in each process:
the warp count changes how `tl.sum` combines the row. In the MI355X QA audit the initial candidate, an exact copy of
the baseline, matched it bit for bit on every case except n4100, where the two processes tuned to different
configurations: `mean` exceeded the zero-violation default on 0.15% and 0.22% of the rows (peak 2.1x and 3.9x of
the bound, bias z-score at most 0.01) and `rstd` reached 0.83x and 0.96x of it. A larger `k` would be needed to
cover a 3.9x peak and would loosen every row; allowing 1% of the rows over the bound accepts the run-to-run
reduction order with about a 4.5x margin over the worst observed fraction. The one- and three-row cases admit no
violation, and a wrong result that moves most rows (a shifted, partially written or fp8-rounded output) still
exceeds the 1% budget.

`rstd` is a float32 `rsqrt` of a float32 variance. A float32 evaluation of the reference (the QA calibration probe)
has a small systematic rounding bias against the float64 reference: reproduced on the CPU with 1M rows, the mean
relative error is +6.5e-9 against an RMS of 4.3e-8 (bias z-score 0.15, about 1/6 of an RMS), and a candidate that
computes `1/sqrt` or the centered variance in float32 shows 0.11 to 0.14. The baseline-relative bias test allows
only 5/sqrt(n) of an RMS, so on the large cases a correct float32 candidate failed it although its peak error was
at most 1.0x of the bound and no row was over it. The relative-error bound does not depend on the baseline's
bias: `rstd` is accepted when it is within `rtol` = 3e-7 (5 float32 unit roundoffs, plus one more from the
comparator) of the float64 value. Measured float32 evaluations peak at 1.4e-7 (torch var), 1.7e-7 (`1/sqrt`),
1.9e-7 (centered) and 2.1e-7 (E[x^2] - mean^2) in 1M rows, all inside the bound, with no violation allowed. The
`shifted_16_ulp` control moves every row by 9.5e-7 (2.6x the limit), and a zeroed, partially written or fp8-rounded
`rstd` is far outside it.

## Workloads

A row moves about 1 KB, so a case needs roughly 30k rows to exceed the 10 us timing floor: performance cases
start at 32768 rows and span 128x up to 4.2M rows (4.3 GB moved); the float64 reference works in chunks. The
original task's other widths (128 to 512) and its unweighted variant are not part of this scenario; smaller batches
are correctness-only.
