# Top-k selection, float32, vocabulary 128256, k = 8

For each row of `x` of shape `(n_rows, 128256)` float32, return the 8 largest values in descending order
(`out_values`, float32) and their column indices (`out_indices`, int64). This is the top-k step of LLM
sampling over a Llama-3-sized vocabulary.

Baseline: the Triton top-k of the earlier `geak_eval/L2/topk` arena task, derived from AITER's Triton top-k
(`aiter/ops/triton/topk.py`, itself adapted from FlagGems). Rows up to 8192 elements use a one-stage
kernel; longer rows (every case here) use a two-stage kernel (per-chunk top-k, then a bitonic merge). The
dispatch tables are those of the earlier task. Changes: the benchmark harness was removed and the
dispatchers write into preallocated outputs. MIT license, copied to `baseline/LICENSE`.

Scenario choice: the earlier task swept 80 `(batch, row length, k)` combinations (rows of 16 to 128256,
k of 2 and 8). This task fixes the production-size row (128256) and k = 8 and varies the batch. Other row
lengths and k = 2 would be separate scenarios.

Inputs: a task generator places the normal quantiles `ndtri((j + 0.5) / n)` at random columns of each row,
so rows follow a standard normal distribution but every value is distinct. Both outputs use the `exact`
bound (a selection returns input values verbatim and indices are discrete), which departs from the
default; exactness is only meaningful without ties, hence the distinct inputs. Tie-breaking is therefore
not exercised: with ties any valid index choice would be acceptable, which `exact` would reject.

Workloads: batches of 1 to 8 (AITER's top-k test sizes) run under the 10 us timing floor and are
correctness-only, as are 100 and the odd sizes; 16 to 4096 rows are timed. GPU audit risk: the
smallest timed batches (16, 64) may measure near 10 us on MI355X; drop them to correctness-only if so. The
float64 reference of the largest case needs about 4 GB.

## Baseline adaptation

The vendored `baseline/kernel.py` dispatcher referenced the module constant `MAX_TINY_ROW` (the one-stage versus
two-stage row-length threshold, 8192 as in the earlier arena task) without defining it, so every case raised a
`NameError`. The constant is now defined next to the other lookup tables; the algorithm is unchanged. Rows of
128256 elements always take the two-stage path.
