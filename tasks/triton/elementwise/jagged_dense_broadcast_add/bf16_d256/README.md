# Jagged + dense broadcast add, bfloat16, width 256

`out[r, :] = jagged[r, :] + dense[b, :]` for every row `r` of sequence `b`, where sequence `b` owns rows
`seq_offsets[b]` to `seq_offsets[b + 1]` of a jagged tensor. This is the per-sequence bias broadcast of
HSTU-style generative recommenders: a variable-length batch of 64 sequences with 256-wide bfloat16
embeddings and one dense row per sequence. The varying axis is the total jagged length.

## Inputs

`max_seq_len` is a scalar input, as in the production call; the launcher uses it to size its grid.
`seq_offsets` comes from the task generator in `inputs.py`: sequence lengths proportional to weights in
[0.5, 1.5) that sum exactly to `total_rows` and never exceed `max_seq_len`. Every case sets
`max_seq_len` to twice the mean length, so a candidate cannot assume uniform lengths but must handle
sequences shorter than one tile and an unused tail of the grid.

## Reference and comparison

The reference adds in float64, gathering each row's dense vector with `repeat_interleave` over the
sequence lengths. A bfloat16 add of two bfloat16 values has a single rounding, so the default
`baseline_relative` bound applies unchanged.

## Baseline

The Triton kernel `jagged_dense_broadcast_add_kernel` of the generative-recommenders project
(`generative_recommenders/ops/triton/triton_jagged.py`, Apache-2.0), forward only, together with its
autotune configurations and launcher, taken from this repository's earlier
`triton2flydsl/generative_recommenders/jagged_dense_broadcast_add` task (the provenance revision is that
repository commit; the upstream commit was not recorded there). Changes: the launcher writes into the
preallocated `out` instead of allocating, the `generative_recommenders.common` helpers are inlined as
a plain `triton.autotune`, and the output pointers use the output row stride (the original used the
jagged stride, which is the same value for contiguous tensors).

## Workloads

Performance cases run 40,960 to 4,194,304 rows (102x); each case moves 1 KiB per row (read plus write),
about 42 MB at the smallest, above the 10 us floor. The 4 M-row case holds a 2 GiB input and output.
Tiny and unaligned totals (1, 3, 100, 1000, 33333) are correctness-only; with 64 sequences, totals
below 64 leave empty sequences, which the kernel must skip.

## Notes for the audit

The autotune key is constant (`STATIC_MAX_SEQ_LENS` is empty in the source), so the configuration is
tuned once on the first launch, which must happen before graph capture.
