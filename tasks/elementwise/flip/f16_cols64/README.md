# Row flip, float16, 64 columns

Reverses each row of `x[N, 64]` (`out[i, j] = x[i, 63 - j]`); a pure data-movement operator.

- **Origin.** The `flip_kernel` of the ROCm Triton benchmark suite (`test_triton_flip.py`), whose
  old tests flip a single tile for int32, float16, bfloat16 and float32. This task fixes float16 with 64
  columns; the siblings `f32_cols64`, `f16_cols64`, `bf16_cols64` and `i32_cols64` cover the other
  dtypes of the old suite. The upstream revision was not recorded.
- **Baseline.** The original kernel (`tl.flip` along columns) generalized so the grid runs over tiles
  of 64 rows; the original handled one tile per launch. The old input/output aliasing test is dropped:
  the task passes distinct buffers.
- **Workloads.** Performance cases run from 262144 to 33.6M rows (33 MB to 4.3 GB per tensor), spanning 128x in rows.
  Smaller shapes are correctness-only.
- **Inputs.** `inputs.py` plants signed zero, +/-inf and the largest finite float16 value at the start of the tensor, as the old tests did. NaN is omitted
  because outputs are poisoned with NaN, so it could not tell a copied element from an unwritten one.
- **Reference.** The reference flips the float64 upcast of the input.
- **Comparison.** `exact`; flipping must preserve every bit. This departs from the default bound
  because the operation only permutes values; no numerical approximation exists to allow for.
- **Dropped coverage.** The old suite also tried other widths (1 to 512 columns, non-power-of-two); this
  scenario fixes 64 columns, and those widths would be separate scenarios.
