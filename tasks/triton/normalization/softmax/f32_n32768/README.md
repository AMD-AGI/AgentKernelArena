# Softmax, float32, 32768 columns

Row-wise softmax along the last axis of an `[n_rows, 32768]` float32 tensor, numerically stabilised by
subtracting the row maximum. Output is float32.

**Production origin.** KernelBench level 1 "Softmax" (float32 logits, row softmax; its suite uses
2048 and 4096 rows at 32768 columns, and also 16384 and 131072 columns, which are other scenarios).
The baseline is AITER's online-softmax Triton kernel (ROCm/aiter `aiter/ops/triton/softmax.py` and
`_triton_kernels/softmax.py`, MIT, hence `baseline/LICENSE`). One program handles a full row in two
block-strided passes (`BLOCK_SIZE` = 16384 elements here).

**Workloads.** Performance cases span 128 to 16384 rows (128x). The two original KernelBench row counts
(2048, 4096) are included. Inputs are `normal(0, 2)` logits instead of KernelBench's `rand`, so rows have
a pronounced maximum and an unshifted exponent would visibly lose accuracy. Single rows, odd and
unaligned row counts are correctness-only boundary cases. The 16384-row case moves 4 GiB.

**Comparison: `sqnr` with `min_db: 128`** (an override, the same bound for all seven float32 softmax widths, 1024 to
131072 columns). A float32 softmax carries two kinds of error: each element's rounding of `x - max` and of the
exponential (a relative error that grows with `|x - max|`), and one shift per row from the float32 rounding of the
row sum, shared by every element of the row and dependent on the summation order. The `baseline_relative` bound
suits neither. Its bias test counts every element as an independent sample, so a correct summation order other than
the baseline's reads as a bias at random, and its elementwise test rejects exponentials that round differently from
the baseline's. Under the current comparator, the previous override (`k: 4`, `max_violation_fraction: 0.001`)
rejected, in a CPU emulation over the seven widths, a Triton-style kernel (`tl.exp` lowered as `exp2(x * log2e)`,
approximate division) in 19 of 68 draws (7 of 8 at 1024 columns, with 0.10% to 0.16% of the elements over the
bound), CPU `torch.softmax` in 32 of 36 draws at 16384 columns and more (0.12% to 1.2%), and the float32 evaluation
of the reference in 6 of 68 draws (bias test only).

`sqnr` compares the norm of the whole output's error with the norm of the reference rounded to float32, with no bias
test and no per-element rule. Evidence for this width (32768 columns): a CPU float32 emulation with `normal(0, 2)`
logits, scored by the evaluator's own `compare_tensor`; 4096 rows in all, the smallest performance case (`r128`)
evaluated as 32 independent draws, and every row also scored alone (the single-row case `r1`). The baseline is
modelled operation by operation (AITER kernel: 16384-wide blocks with the online rescale, each `tl.sum` as 8 warps x
64 lanes x 32 sequential values and a shuffle tree, `tl.exp` as `exp2(x * log2e)`, a reciprocal multiply for the
division), with `exp2` and the reciprocal either correctly rounded or randomly off by up to one ulp (a model of the
hardware units).

| Implementation (float32) | `r128`, worst draw (dB) | single rows: lowest (dB), share below 128 dB |
| --- | --- | --- |
| Baseline model, correctly rounded / 1-ulp units | 139.0 / 138.3 | 132.2 / 131.5, 0 |
| `torch.exp`, pairwise sum, division (the QA float32 probe) | 141.5 | 134.8, 0 |
| Triton-style: `exp2(x * log2e)` on a 1-ulp unit, tree sum, reciprocal multiply | 136.9 | 132.1, 0 |
| One 64-lane wavefront per row: `fma` exponent argument, 1-ulp `exp2`, lane-sequential sums | 136.1 | 130.4, 0 |
| Online softmax over 1024-wide blocks (block sums added sequentially) | 137.7 | 129.5, 0 |
| CPU `torch.softmax` (16 sequential vector lanes per row sum) | 132.4 | 123.9, 2.7e-2 |
| Control `shifted_16_ulp` (whole output / highest single row) | 120.8 | highest 123.2, 1 |
| Control `fp8_rounded` (whole output / highest single row) | 8.5 | highest 23.2, 1 |

`min_db: 128` separates the two groups at every width. Every correct design in the table, except CPU `torch.softmax`
at 131072 columns (see below), passes every performance case of every width, by at least 6 dB (2x in noise
amplitude) except CPU `torch.softmax` at 16384 and 32768 columns (4.4 to 4.5 dB) and the 1024-block online softmax
at 131072 columns (4.3 dB). No single row of the baseline model or of the float32 probe falls below 128 dB; one row
of the Triton-style kernel (127.3 dB, 1 of 65536 at 1024 columns) and one of the one-wavefront kernel (127.6 dB, 1
of 16384 at 8192 columns) do. The `shifted_16_ulp` control scales the output by `1 + 16 u` (u = 2^-24), so its SQNR
cannot exceed about 120.4 dB plus the baseline's own error: whole outputs measure 120.5 to 120.9 dB (noise 2.1 to
2.4 times the allowance) and no single row exceeds 124.0 dB, so it fails every case. `fp8_rounded` stays below 40
dB, zeroed, half-written and stale outputs at about 3 dB or less, and one zeroed row in a case of a million rows at
60 dB.

Not covered. A 5x margin on `shifted_16_ulp` (`min_db` of about 135 dB) would reject single rows of the baseline
model (lowest 130.0 dB) and of the float32 probe (lowest 128.7 dB). CPU `torch.softmax` is limited by its row sum,
not by its exponential: `torch.exp` with the same 16-lane sequential sum reproduces it within 0.7 dB, while
`torch.exp` with a pairwise sum is the probe row. Its row-sum error grows with the width (RMS 1.0 u at 1024 columns,
3.0 u at 32768, 6.0 u at 131072); at 131072 columns a quarter of its rows fall below 128 dB and its worst rows reach
118.5 dB, below the 120.8 dB of the `shifted_16_ulp` control, so no bound that rejects that control can accept it
there. The online softmax over 1024-wide blocks adds 128 block sums one after another at 131072 columns; 2% of its
single rows fall below 128 dB, so it fails `r1` in a few percent of input draws. SQNR does not localise errors: a
lone element wrong by a small relative amount in a large case is not detected.

**Notes.** The AITER kernel repr helper and logger were removed. The sibling HIP task `SoftmaxModule` (tiny 3D cubes
of side 4 to 64, softmax over the last axis) is a different width and every one of its shapes is far below the 10 us
floor, so it is not carried over.
