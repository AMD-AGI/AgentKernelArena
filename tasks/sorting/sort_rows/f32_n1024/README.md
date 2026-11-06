# Row sort, float32, 1024 elements per row

`out[r, :] = sort(x[r, :])` ascending, for `x` of shape `(n_rows, 1024)` float32. The baseline is the
`tl.sort` bitonic-sort Triton kernel of the old `rocmbench/medium/test_triton_sort` task, one program per row.

Scenario choice: the old task swept int32/float16/float32/bfloat16, ascending and descending, and tile
shapes up to `(256, 64)`; its kernel launched a single program over one tile. This task fixes the common
float32, ascending, 1024-element-row case and varies the number of rows. The baseline kernel gains a
program-id offset (the only change to the kernel body) so that it handles many rows.

Comparison is `exact`: a sort only permutes input values, so the output must equal the float64 reference
cast back to float32 bit for bit (float32 inputs are exactly representable in float64). This departs from
the default bound by design.

Inputs: a task generator draws standard-normal rows; odd rows draw each element from a per-row pool of
n/8 full-precision normal float32 values, so they contain many duplicate values (the old task sorted
integer-valued data in [0, 100), also heavy in ties). Values are deliberately not rounded to a coarse grid:
a grid such as multiples of 1/8 is exactly representable in bfloat16 and fails the QA-INPUTS check.
The old task's other dtypes (int32, float16, bfloat16) and descending order are not covered here: a
scenario fixes one dtype and this one fixes float32 ascending; they would be sibling scenarios.

Workloads: the old suite had no batched production sizes; the row counts are scaled batches of its
1024-element rows. Counts below 256 (about one row per compute unit) run faster than the 10 us timing
floor and are correctness-only. Risk for the GPU audit: if the baseline is faster than 10 us at 256 rows,
that case must become correctness-only.
