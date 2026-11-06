# Row gather (index_select along dim 0), float32, 4096 columns

`out[i, :] = table[idx[i], :]`: the embedding-lookup / `torch.index_select(table, 0, idx)` pattern.
The scenario is a 32768 x 4096 float32 table gathered by a varying number of int64 indices
(drawn uniformly, so duplicates occur). Correctness is bitwise (`exact`): a gather copies values
and any deviation is a bug, so the default relative bound would only loosen the check.

Origin: the old task wrapped the GPU-mode `Gather` module (`input[indices]` on tiny 1-D/2-D/3-D
toy tensors, all far below the timing floor). It is generalized here to the production use of
that operator, a row gather from an embedding table. The row width 4096 and 32768 rows are an
assumption (a mid-size LLM hidden width and vocabulary); they were chosen so the case sizes move
enough bytes (2048 rows = 32 MiB read + 32 MiB written) to exceed the 10 us floor. Batches
under about 2048 rows are correctness-only.

Baseline: a standalone Triton rewrite of the indirect-load pointwise kernel TorchInductor emits
for `index_select` (PyTorch, BSD-3-Clause); there is no vendored production Triton source to
copy. One program per (row, 1024-column block).

The `exact` comparison departs from the default bound deliberately: the operation is a pure copy
(gather), so any difference from the reference is a defect.
