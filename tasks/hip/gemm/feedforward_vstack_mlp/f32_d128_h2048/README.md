# Stacked two-layer MLP with sigmoid output, float32, 128 -> 2048 -> 1

`out = sigmoid(relu(vstack(x, y) @ fc1_weight^T + fc1_bias) @ fc2_weight^T + fc2_bias)`. `x` and `y` are
`[n_rows, 128]` float32 matrices stacked vertically, so the output has `2 * n_rows` rows of one value each;
rows `0..n_rows-1` come from `x`, the rest from `y`. The stacked rows are independent, so the GEMM is
`[2 * n_rows, 128] x [128, 2048]` followed by a 2048-term dot product per row.

- **Merged sources.** `hip2hip/Feedforward` and `torch2hip/gpumode/10024_Feedforward` are the same operator
  (gpumode 10024) with `input_size=128`, `hidden_size=2048`. `x` and `y` always have the same row count here;
  the old workloads never used unequal counts.
- **Inputs.** `fc1` uses PyTorch's default `Linear` initialization (uniform within `+-1/sqrt(128)`). `fc2` is
  normal with a standard deviation of 0.05 and the bias with 0.1, wider than the default (`+-1/sqrt(2048)`): with the
  default the sigmoid outputs of all rows lie within about 1% of 0.5, which would make a constant or row-independent
  answer pass. With these values the pre-sigmoid logit has a standard deviation of about 0.8 across rows, so the
  outputs spread over most of (0.1, 0.9).
- **Baseline.** The provided HIP kernel of `hip2hip/Feedforward` (task-original, not an upstream production
  kernel), restricted to float32: one thread per stacked row caches the row in registers and loops over the
  hidden units in tiles of 32, with `fc1_bias`, `fc2_weight` and `fc2_bias` in shared memory and `fc1_weight`
  read from global memory. The host-side `contiguous()` copies and `torch::empty` of the old launcher are gone:
  the kernel writes the preallocated output. The register row cache limits `in_features` to 256.
- **Workloads.** Per-input row counts 512 to 4096 are the gpumode cases; 64 to 256 and 8192 to 16384
  extend them to the 100x span. The baseline gives one thread (not a wave) all the work of a row, a
  `~0.5 MFLOP` sequential dependent loop of 262k FMAs, so even the smallest case takes far longer than 10 us
  (latency-bound), and larger ones scale until the GPU is full. Row counts 1, 3, 48 and 1000 are correctness-only
  boundary cases.
- **Comparison.** `sqnr` with `min_db: 128` instead of the default `baseline_relative`. The baseline is a
  plain sequential-FMA fp32 kernel whose error is close to one output rounding, so a per-element bound of
  `k` times that error rejects correct fp32 candidates that only accumulate the 128-term and 2048-term
  reductions in a different order (split-K, matrix instructions): a float32 PyTorch evaluation of the
  reference violated the bound on 2% to 17% of elements (peak up to 12x the bound, no bias) in every case. That is
  legitimate float32 accumulation noise. Measured on CPU with the declared input distributions (30 seeds per
  size, 1 to 1000 rows per input): float32 evaluation has SQNR 130.7 dB at worst (median 137 dB, the smallest
  outputs being the noisiest) against the float64 reference; splitting the reductions 2, 4 and 8 ways gives
  138 to 141 dB at 2048 rows. A relative output shift of 8 and 16 unit roundoffs gives 126 and at most 122 dB;
  TF32 and bf16 inputs give 70 and 51 dB. The 128 dB floor accepts float32 accumulation and rejects the
  16-ulp shifted control, low-precision (TF32, fp16, bf16, fp8) results, zeros, stale or half-written outputs.
- **Risk.** The HIP build and timing are not verified on a GPU. Because the baseline is latency-bound for small
  batches, speedups on the small cases may be very large; the largest cases bound the stacked batch at 32768 rows.
