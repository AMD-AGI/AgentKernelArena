# Back-to-back GEMM, float16, head size 64

`e = (a @ b^T) @ c` for `a` [n_rows, 64], `b` and `c` [n_keys, 64], all float16. The [n_rows, n_keys]
intermediate never reaches memory: this is the attention pattern (scores, then weighted values) without
the softmax. Both products accumulate in float32; the baseline rounds the intermediate to float16 between
them, so the reference (float64, no intermediate rounding) is judged with the default baseline-relative bound,
which absorbs that rounding.

**Origin.** Three rocmbench tests exercised a dot consuming another dot (`test_gemm_fusion`,
`test_chained_matmul`, `test_chained_dot_fp8`). This task takes the float16 `test_gemm_fusion` kernel; the
chained-matmul variant is the same operator with a different layout, and the FP8 chained dot (an fp8
sibling scenario) is not migrated here. Inputs keep the original distribution (normal, mean 0.1, std 0.2);
with up to 16384 keys the outputs reach about 1e3, well inside float16.

**Workloads.** The original tests were tile sweeps up to 4096 x 4096; these cases extend the same chained
product to attention-like sequence lengths (up to 16384 x 16384) so each timed case is compute-bound
(performance cases span 128x in `n_rows * n_keys`). Small and odd shapes are correctness-only.

**Baseline.** The original kernel with block pointers (which need whole tiles) replaced by masked pointer
loads and stores so any size works (zero padding contributes nothing to either dot). One program handles 128
rows and streams B and C in 128-row tiles; with few rows the grid is small, which is why the baseline is
slow at small `n_rows` and leaves room for splitting the key range. License header: see `baseline/LICENSE`.

**Coverage notes.** The old tests also ran head sizes 32 and 128 (an exact 0/1-valued `test_chained_matmul`
and a head-size sweep); this scenario fixes head size 64, the size the old `test_gemm_fusion` used for its
4096 x 4096 case, and head size is a `const` axis, so other sizes would be sibling scenarios. The small
and odd `n_rows` / `n_keys` correctness cases (1, 3 x 5, 100 x 130, 1000 x 333, 257 x 4100) cover the
masked-tail paths that the old whole-tile block pointers never exercised.
