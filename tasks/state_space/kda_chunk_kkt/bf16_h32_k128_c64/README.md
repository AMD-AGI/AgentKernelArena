# KDA gated chunk K K^T and Q K^T, bf16, 32 heads, K=128, chunk 64

The intra-chunk matrices of Kimi Delta Attention (Kimi-Linear) chunked prefill. For every chunk of
64 tokens and every head, with per-channel cumulative log2-decay gates `g`:

```text
A[i, j]   = beta[i] * sum_k k[i, k] * k[j, k] * 2^(g[i, k] - g[j, k])   for i > j,   0 otherwise
Aqk[i, j] = scale   * sum_k q[i, k] * k[j, k] * 2^(g[i, k] - g[j, k])   for i >= j,  0 otherwise
```

`A` and `Aqk` are `[T, 32, 64]` float32; everything above the diagonal (and, for `A`, the diagonal)
is exactly zero. `q` and `k` are `[T, 32, 128]` bfloat16, `beta` `[T, 32]` bfloat16, `g`
`[T, 32, 128]` float32, `scale = 1/sqrt(128)`. One sequence batch (no variable-length metadata).
The matrix `A` later goes through a triangular solve, which is a separate operator.

- **Origin.** `chunk_kda_scaled_dot_kkt_fwd` of flash-linear-attention as carried in vLLM
  (`vllm/third_party/flash_linear_attention/ops/kda.py`), which launches two kernels: `sub_inter`
  (the 16x16 sub-blocks strictly below the diagonal, MFMA dots) and `sub_intra` (the diagonal
  sub-blocks, column by column). The task replaces the two earlier tasks that covered one half each
  (with 32-wide float32 test data); the dimensions follow Kimi-Linear (32 heads, head size 128).
- **Inputs.** `q, k ~ N(0, 1/128)` (as for L2-normalized vectors), `beta ~ U(0, 1)` (a sigmoid
  output), and `g` from `inputs.py`: per-token, per-channel log2-decays in `[-0.25, 0]` cumulated
  inside each 64-token chunk, so `g` is non-increasing and the decay factor is at most 1 below the
  diagonal, as in production (where `g` is the chunk-local cumulative sum scaled by 1/ln 2).
- **Baseline.** Both vLLM kernels, single sequence (the varlen path and the batch axis removed),
  autotune kept. One deliberate change: production clears `A` and `Aqk` with `torch.zeros` before
  the kernels; because the evaluator poisons the outputs and a call may not run PyTorch compute,
  the `sub_inter` kernel itself stores zeros into the sub-blocks on and above the diagonal (every
  output element is then written; the `sub_intra` kernel overwrites the diagonal sub-blocks after
  it on the same stream). The beta-scaled key in `sub_intra` is rounded to bfloat16 as in vLLM, so
  the baseline's `A` has about 57 dB SQNR against the float64 reference; its `Aqk` has no
  intermediate rounding (gates rebased per 16-row sub-block, so every decay factor is at most 1)
  and has about 139-140.5 dB.
- **Workloads.** 1024 to 131072 tokens (128x span). Traffic is about 48 KB per token (q and k 8 KB
  each, g 16 KB, both outputs 16 KB), so the smallest case is about 50 MB and should take the
  baseline over 10 us. Production prefill chunks are typically 2048-16384 tokens per step; larger
  counts stand for batched or long-context prefill and are needed for the span. Axis minimum 1,
  17, 65 (a partial chunk) and 1000 (non-power-of-two, not a multiple of 16) are correctness-only.
- **Comparison.** The reference is float64 and evaluates `(q * 2^g) @ (k * 2^-g)^T` per chunk,
  which stays well inside float64 range for this gate distribution (output about 4.3 GB per tensor
  at the largest case, inputs about 6 GB). Both outputs are float32, their error comes from float32
  accumulation and the gate exponentials rather than output rounding, so the default bound (twice
  the baseline's own error or its row RMS) compares two independent rounding errors element by
  element and rejects equally accurate implementations. On MI355X, under the comparator's earlier
  row rule (RMS over all 64 columns), a float32 PyTorch evaluation of the reference failed it on
  0.11% of the `A` and 4.6% of the `Aqk` elements (peaks 32x and 3500x the bound). Under the
  current rule (row RMS over the elements that are nonzero in the reference or the baseline, rows
  with fewer than 8 of them on the whole output's RMS) a CPU float32 emulation still fails it for
  each of five correct designs (the first four below and the other `BK`), on 1.0e-4 to 1.6e-3 of
  the `A` and 6.9e-4 to 4.4e-2 of the `Aqk` elements (1024 and 4096 tokens, two draws, peaks up
  to 18x). The emulation models the baseline
  with its blocking and roundings (both autotune choices, `BK` 64 and 32) and compares correct
  float32 designs against it: the float32 evaluation of the reference (which on the CPU equals
  gate-scaled float32 operands with a sequential FMA dot), one 64x64 GEMM per chunk with gates
  rebased at the chunk start, a per-pair decay `2^(g_i - g_j)`, gate-scaled operands with 4-term
  MFMA groups, the same without rebasing and with `2^x` evaluated as `exp(x ln 2)` (4-term groups
  and an FMA dot), the baseline algorithm with `exp(x ln 2)`, with a float32 `beta * k` or with
  `beta` applied after the dot in the diagonal sub-blocks, and the other `BK`.
  - `Aqk`: `sqnr` with `min_db: 127`. Correct designs reach 132.9 to 140.4 dB (least accurate:
    the unrebased `exp(x ln 2)` FMA design 132.9 dB, the rebased single GEMM 133.7 dB, the float32
    reference evaluation 134.0 dB; the baseline about 139 to 140.5 dB). The evaluator's 16-ulp shift
    control sits at 120.5 dB (20 log10 2^20) and fp8 rounding at 16 dB, so 127 dB leaves 5.9 dB
    below the least accurate correct design and 6.5 dB above the shift.
  - `A`: `baseline_relative` with `k: 8` and `max_violation_fraction: 6.0e-4`. No single SQNR
    threshold works here, because the baseline's bfloat16-rounded diagonal sub-blocks keep it at
    57 dB, below the 16-ulp shift. The per-element bound stays bfloat16-level in the rows that
    hold diagonal-sub-block entries and is float32-tight in the rows that hold only `sub_inter`
    entries (rows 16, 32 and 48 of every chunk) and in the rows whose `beta` is a power of two
    (there `beta * k` is exact in bfloat16). All violations of correct designs fall in those rows
    (a few elements, at most 1.4x the bound, in rows with one or two diagonal-sub-block entries).
    - Performance cases (1024 to 131072 tokens, two draws each for the float32 reference
      evaluation, the rebased GEMM, the per-pair decay, the 4-term gate-scaled design, the
      `exp(x ln 2)` baseline and the other `BK`; the other designs up to 4096 or 16384 tokens, and
      the unrebased `exp(x ln 2)` FMA design also with one draw of 131072): at `k = 8` the correct
      designs violate on at most 1.43e-4 of the elements (that FMA design; at most 9.7e-5 for the
      others) with a peak of at most 5.4x the bound (the float32 reference evaluation and the
      rebased GEMM at 131072 tokens).
    - Correctness cases (40 draws of 17 and of 65 tokens, 8 of 1000, both baseline models): at
      most 1.95e-4, peak at most 4.4x.
    - Controls: the 16-ulp shift violates on 2.7e-3 to 4.1e-3 in every performance case (at least
      1.3e-3 in the correctness cases); fp8 rounding on at least 0.40 of the elements, with peaks
      above 1e6x.
    - Choice: `k = 8`, the largest allowed value, because the peak shrinks with `k` (6.5x at
      `k = 6`, 8.8x at `k = 4`, 17x at `k = 2`) while the shift control stays rejected.
      `max_violation_fraction: 6.0e-4` is 4.2x above the worst correct design in the performance
      cases (3.1x above the worst correctness-case draw) and 4.6x below the lowest violation
      fraction of the shift control in a performance case; the 5.4x peak leaves 3x to the
      comparator's 16x limit for admitted elements.
