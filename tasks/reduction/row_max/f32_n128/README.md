# Row maximum, float32, 128 columns

`out[m] = max_n x[m, n]` for `x` of shape `(M, 128)` float32, with a float32 result.

Correction to the migration plan: the old task `triton2triton/rocmbench/easy/test_load_reduce` is
named like a sum but its kernel (`load_reduce_kernel`) loads a block with `tl.make_block_ptr` and
computes `tl.max` over axis 1; the scenario is therefore a row maximum, not a row sum, and the task
is `row_max`. The old kernel reduced one fixed `(128, 64)` block in a single program (grid `(1,)`),
and its performance cases (`128..1024` rows) took well under 10 us. This task generalizes the same
kernel over rows: each program handles 128 rows, and the timed cases use millions of rows so that
the baseline moves 34 MB to 3.4 GB. The column count is fixed at 128, one of the original
performance widths.

Bound: `exact`. The maximum of float32 values is a float32 value, so any correct kernel matches the
float64 reference rounded to float32 bit for bit; inputs are normally distributed and never NaN.

Baseline: the original kernel (block-pointer load, `tl.max`, store), extended with a row offset per
program, a boundary check on the row axis and a masked store, wrapped in `run()`. The original
source is AMD-authored and carries no upstream license, so provenance is the task itself
(`task-original`).

Risk: the smallest timed case (65536 rows, 34 MB) is estimated at 11-12 us on the bandwidth model;
check on hardware that the baseline takes at least 10 us there.

Sibling scenario of `fp16_n128`: the original task benchmarked this kernel in float16, float32 and
bfloat16, so each dtype is its own scenario with identical structure.
