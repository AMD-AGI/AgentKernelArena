# BF16 GEMM, N=1280, K=8192

`out = x @ w.T` with `x` of shape `[m, 8192]` and the row-major `[1280, 8192]` weight `w`
(the `nn.Linear` layout), all bfloat16, accumulated in float32 and rounded to bfloat16. No bias and no
activation. The var axis is `m`, the number of rows (tokens).

## Production origin and workloads

The shape is the fused QKV projection of Llama-3-70B at tensor-parallel degree 8 (1280 output
features, hidden size 8192). Cases span `m = 1 .. 16384`: decode batches and the smallest chunks
(`m <= 100`) are correctness-only, `m = 128 .. 16384` are performance cases (128x span). The three
boundary correctness cases are `m = 1`, `m = 33` (not a power of two, not a multiple of 16) and
`m = 100`. Both inputs are standard normal.

## Baseline

`baseline/kernel.py` is a Triton GEMM following AITER's `gemm_a16w16` (MIT, so `baseline/LICENSE` is
included): grouped tile order, one output tile per program, float32 accumulation, and a small tile table
(64x64x32 tiles, 32-row tiles for `m <= 32`, 128x128x64 only when both `m` and `n` are at least 2048,
which never holds at `n = 1280`). It is a task-local simplification of the AITER function, not a verbatim
copy: AITER's tuned per-shape configuration files are not used, and the optional bias and activation
epilogues of the earlier task were dropped. It is intentionally not heavily tuned. The
split-K atomic variant of the earlier task is not included.

## Comparison

`baseline_relative` (default bounds). The reference is float64 `x.double() @ w.double().T`.

## Risks for the GPU audit

Real baseline times at `m = 128` are expected to be well above 10 us because only 40 programs are
launched at 64x64x32; this has not been measured.
