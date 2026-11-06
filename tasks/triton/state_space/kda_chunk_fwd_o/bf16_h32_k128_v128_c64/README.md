# KDA chunk output pass, bfloat16, 32 heads, K = V = 128, chunk 64

The last kernel of Kimi Delta Attention (KDA) chunked prefill (`chunk_gla_fwd_kernel_o`). For each
chunk of 64 tokens and head, it combines the state at the start of the chunk with causal attention
inside the chunk:

```text
o_i = scale * (q_i * 2^g_i) @ h_c^T  +  sum_{j <= i in the chunk} A_ij * v_j
```

`q` `[T, 32, 128]`, `v` `[T, 32, 128]`, `h` `[ceil(T/64), 32, 128 (v), 128 (k)]`, `A` `[T, 32, 64]`
and `o` `[T, 32, 128]` are bfloat16, `g` `[T, 32, 128]` float32, `scale` = 128^-1/2. One sequence
(no variable-length metadata); the last chunk is partial when `T` is not a multiple of 64.

- **Origin.** vLLM serves Kimi-Linear (and Kimi-K3) KDA layers on ROCm with the kernels in
  `vllm/models/kimi_k3/amd/ops/third_party/kda/chunk.py`; this is their output pass, with
  Kimi-Linear's 32 heads of size 128. In that pipeline `q` is the L2-normalized query, `v` the
  delta-rule corrected values and `h` the chunk states of the state pass, `g` the gate's
  chunk-local cumulative log2-decay and `A` the bfloat16 `Aqk` of the intra-chunk pass.
- **Baseline.** That vLLM kernel, unchanged except that its `@triton.autotune` (key `BT` only, so
  the configuration would be tuned on whichever case runs first) is replaced by one entry of its
  configuration space (BK 64, BV 64, 4 warps, 2 stages); the wrapper serves one sequence and writes
  into the preallocated `o`. The earlier task of this operator used an older flash-linear-attention
  variant of the kernel (state stored `[K, V]`, natural-log gates, every tensor float32, one
  64-token shape with 2 heads of size 32); this task follows the production kernel and dtypes.
- **Inputs.** `q ~ N(0, 1/128)` (as for L2-normalized queries), `v ~ N(0, 1)`, `h ~ N(0, 0.5^2)`.
  `inputs.py` draws `g` as the production gate would: per-token safe-gate decays
  `-5 * sigmoid(exp(A_log) * (x + dt_bias))` with per-channel parameters spanning no decay to fast
  decay, divided by ln 2 and summed within each chunk. `A` is N(0, 0.01^2) on and below the
  diagonal; above it the producer leaves the matrix largely unwritten, so those entries are
  arbitrary values U(-1, 1) that the operator must not read (reading them changes `o` by about 50x).
  The state term carries about half of the output RMS.
- **Reference.** Float64, chunks processed in groups of 64 without intermediate rounding, checked on
  CPU against an independent per-token loop. The baseline in the Triton CPU interpreter has about
  54 dB SQNR against it.
- **Comparison.** `baseline_relative` with `k = 4` instead of the default 2. The kernel rounds
  `scale * q` and then `q * 2^g` to bfloat16 before the state dot product, so its error is set by
  those two roundings. An implementation that applies `scale` to the float32 accumulator instead
  rounds only once and is more accurate (55 dB), but its per-element errors are not the baseline's:
  in the CPU emulation it needs `k` = 2.5 at 1000 tokens and 2.7 at 4096 tokens, growing with the
  number of elements, and fails the default bound. Another block configuration (BK 32, BV 128)
  needs 1.2. With `k = 4` the evaluator's controls (16-ulp shift, fp8 rounding) are still far
  outside the bound.
- **Workloads.** 1024 to 131072 tokens (128x span). Traffic is about 60 KB per token (q, v and o 8 KB
  each, g 16 KB, A 4 KB, chunk states 16 KB), so the smallest case is about 61 MB and should take the
  baseline over 10 us. Production prefill chunks are typically 2048-16384 tokens; larger counts
  stand for long-context prefill and are needed for the span. Axis minimum 1, 65 and 100 (partial
  chunks) and 1000 (non-power-of-two, not a multiple of 16) are correctness-only. The CPU pre-check
  smokes only the 1-token case (the largest case exceeds its size limit); the 131072-token case
  needs about 8 GB of inputs and 4.3 GB for the float64 output.
