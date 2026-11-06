# GRU-style gated selection, float32, d_model 512

The `GateGRUSelectionLayer` of the gpumode problem set (problem 5334), used to blend two feature rows
`x1`, `x2` with a GRU-like cell: three `Linear(2*512 -> 512)` projections give a reset gate, an update
gate and a proposal, and `out = (1 - update) * x1 + update * proposal` with
`proposal = tanh(Wp [reset * x1, x2] + bp)`. The old tasks fed 4D `[B0, 4, 4, 512]` tensors; the leading
dimensions are flattened into one `rows` axis, which does not change the operator. Weights and biases use
the `nn.Linear` default bound `1/sqrt(1024)`, so the gates are live (neither saturated nor constant).
The reference is float64 and chunked over rows. The two old task families (hip2hip and torch2hip) wrap the
same gpumode problem and are merged here.

- **Baseline.** Hand-written task-original HIP: one tiled fp32 GEMM kernel computes the reset and update
  gates from a shared `[x1, x2]` tile, a second computes the proposal GEMM over `[reset * x1, x2]` and
  fuses `tanh` and the final blend in its epilogue (64x64x16 tiles, 4x4 register micro-tile per thread,
  no matrix instructions). The old task's provided HIP baseline implemented the three projections with
  `at::linear`, `at::cat`, `sigmoid` and `tanh`, which the HIP candidate rules forbid, so it could not be
  vendored. The reset/update intermediates are allocated inside `run`.
- **Workloads.** Row counts 1 to 65536 (old cases 1024 to 16384 rows, extended to a 256x span).
  Performance cases are 256 to 65536 rows: 0.8 GFLOP of fp32 work at 256 rows, well above the 10 us floor.
  Rows 1, 3, 100 and 1000 are correctness-only boundary cases.
- **Comparison.** `sqnr` with `min_db: 130` instead of the default `baseline_relative`. The baseline is
  a plain sequential-FMA fp32 kernel whose error is close to one output rounding, so a per-element bound
  of `k` times that error rejects correct fp32 candidates that merely accumulate the K=1024 reductions in
  a different order (split-K, matrix instructions, a different tile order): a float32 PyTorch evaluation
  of the reference violated the `k=2` bound on about 3% of elements (peak 5x to 8x the bound, no bias) in
  every case from 1 to 65536 rows. That is legitimate float32 accumulation noise, not a defect, and no
  allowed `k` (at most 8) or `max_violation_fraction` (at most 0.01) accepts it. Measured on CPU at 4096
  rows with the declared input distributions: the float32 evaluation has SQNR 138 dB against the float64
  reference; a 1, 2, 4, 8 and 16 unit-roundoff relative shift gives 136, 135, 131, 126 and 120 dB. The
  130 dB floor therefore accepts float32 accumulation with about 8 dB (2.5x error) of margin and rejects
  the 16-ulp shifted control (120 dB), low-precision (TF32, fp16, bf16, fp8) results (far below 130 dB),
  zeros, stale or half-written outputs.
- **Risk.** The baseline kernels were written without a GPU in reach: HIP build and timing are unverified,
  and the CPU pre-check cannot compile `kernel.hip`.
