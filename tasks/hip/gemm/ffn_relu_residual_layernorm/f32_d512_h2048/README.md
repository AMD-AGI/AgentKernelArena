# Position-wise feed-forward + residual + LayerNorm, float32, d_model 512, hidden 2048

The post-attention sub-block of a Transformer encoder layer (the "Position-Wise Feed-Forward" module of
the gpumode problem set): `out = LayerNorm(W2 relu(W1 x + b1) + b2 + x)`. The block is
evaluated in inference mode, so dropout is the identity (the old task's HIP kernels applied PyTorch's
random dropout with a fixed seed; that cannot be a deterministic kernel contract and is not part of
this task). LayerNorm uses the biased variance, `eps = 1e-5` (the `nn.LayerNorm` default) and an affine
gamma/beta; gamma and beta are drawn away from the 1/0 identity (U(0.5, 1.5) and N(0, 0.1^2)) so a
missing or swapped affine step is visible. Weights and biases use the `nn.Linear` default bound
`1/sqrt(fan_in)`. The reference is float64 and chunked over tokens.

- **Baseline.** The HIP implementation provided by the old `hip2hip/PositionWiseFeedForward` task
  (a task-original kernel, not an upstream production kernel): a 32x32-tile shared-memory fp32 GEMM with
  fused bias+ReLU, a second GEMM with fused bias, and a per-row Welford residual+LayerNorm kernel. It is
  reduced to float32, the pybind module / dtype dispatch / ATen dropout are removed, and the LayerNorm
  kernel writes the preallocated `out`. The two intermediate buffers are allocated inside `run` (a launch
  allocation, allowed by the candidate rules). The baseline is intentionally simple (non-MFMA); a
  candidate that fuses the three stages or uses matrix instructions can be much faster.
- **Workloads.** Token counts 1 to 65536 at d_model 512 / hidden 2048 (the old task used batch x sequence
  of 1024 to 16384 rows; the two larger counts extend that to a 128x performance span). Performance cases
  are 512 to 65536 tokens; the baseline needs at least 2 GFLOP of fp32 work at 512 tokens, far above the
  10 us floor. Tokens 1, 3, 48 and 1000 are correctness-only boundary cases. The largest case holds a
  0.5 GB intermediate and about 1 GB of float64 reference temporaries per chunk.
- **Comparison.** `sqnr` with `min_db: 130` instead of the default `baseline_relative`. The baseline is a
  plain sequential-FMA fp32 pipeline (two K=512 / K=2048 reductions plus LayerNorm), so a per-element bound
  of `k` times its error rejects correct fp32 candidates that merely accumulate in another order (split-K,
  matrix instructions, a different tile order, fused stages): the QA audit's float32 PyTorch evaluation of the
  reference violated the default bound on about 2% of elements (peak 4.4x) in every case, with no bias.
  That is legitimate float32 accumulation noise. Measured on CPU with the declared input distributions
  (1 to 4096 tokens): the float32 evaluation has SQNR 139.5 to 139.9 dB against the float64 reference, a
  re-ordered (chunked-K) float32 evaluation 141 to 143 dB; a 4, 8 and 16 unit-roundoff relative shift of the
  output gives 132.5, 126.5 and 120.4 dB, and TF32-rounded matrix inputs 80 dB. The 130 dB floor therefore
  accepts float32 accumulation with about 10 dB of margin and rejects the 16-ulp shifted control (120 dB),
  low-precision (TF32, fp16, bf16, fp8) results, zeros, stale or half-written outputs.
- **Risk.** HIP build and timing are unverified on a GPU: the CPU pre-check cannot compile `kernel.hip`.
