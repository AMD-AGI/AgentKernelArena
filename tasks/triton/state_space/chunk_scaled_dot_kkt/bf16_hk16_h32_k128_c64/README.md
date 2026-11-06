# Chunk scaled K K^T with gate decay (Gated DeltaNet), bf16, 16 key / 32 value heads, K=128, chunk 64

The intra-chunk matrix of Gated DeltaNet chunked prefill (Qwen3-Next / Kimi-Linear style). For every
chunk of 64 tokens and every value head `h` (key head `h // 2`):

```text
A[i, j] = beta[i] * <k[i], k[j]> * exp(g[i] - g[j])   for i > j,   0 otherwise
```

`out[T, 32, 64]` is float32; the strict upper triangle, the diagonal and the padding rows of a
partial last chunk are zeros. `k` is `[T, 16, 128]` bfloat16, `beta` `[T, 32]` bfloat16, `g`
`[T, 32]` float32 (chunk-local cumulative log-decay). One sequence batch (no variable-length
metadata).

- **Origin.** `chunk_scaled_dot_kkt_fwd` of flash-linear-attention as carried in vLLM
  (`vllm/third_party/flash_linear_attention/ops/chunk_scaled_dot_kkt.py`). It replaces two earlier
  tasks (an SGLang copy and a vLLM copy with 32-wide float32 test data); dimensions follow
  Qwen3-Next's Gated DeltaNet (16 key heads, 32 value heads, K = V = 128, chunk 64).
- **Inputs.** `k ~ N(0, 1/128)` (as for L2-normalized keys), `beta ~ U(0, 1)` (a sigmoid output), and
  `g` from `inputs.py`: per-token log-decays in `[-0.2, 0]` cumulated inside each 64-token chunk,
  so `g` is non-increasing and `exp(g[i] - g[j]) <= 1` below the diagonal, as in production.
- **Baseline.** The vLLM kernel reduced to this scenario: the beta-scaled left operand is formed in
  bfloat16 (as in vLLM) and the dot accumulates in float32, then the decay is applied and the strict
  lower triangle stored. The varlen path, gate-less variant and RDNA switch were removed and the
  autotune replaced by one fixed configuration (K block 64, 8 warps, 3 stages), which also keeps a
  first call from tuning inside a graph capture. Because the baseline rounds `beta * k` to bfloat16,
  its error against the float64 reference is larger than a float32-scaled implementation's (SQNR
  about 55.6 dB) on almost every row; see **Comparison** for the rows where it is not.
- **Workloads.** 4096 to 409600 tokens (100x span). Traffic is about 12 KB per token (4 KB of keys,
  8 KB of output), so the smallest case is about 50 MB and should take the baseline over 10 us.
  Production prefill chunks are typically 2048-16384 tokens per step; larger counts stand for
  batched or long-context prefill and are needed for the 100x span. Axis minimum 1, a partial chunk
  (65) and 1000 (non-power-of-two, not a multiple of 16) are correctness-only.
- **Comparison.** `baseline_relative` with `k: 8` and `max_violation_fraction: 4.0e-4`;
  reference in float64 (output ~6.7 GB at the largest case, inputs ~1.8 GB).
  - *Why the default does not fit.* The baseline's error is bfloat16-level because it rounds
    `beta * k` to bfloat16. The exception is a row whose `beta` is a power of two (0.5, 0.25, ...,
    1.0). There `beta * k` is exact, so the baseline's error drops to float32-accumulation level,
    and twice that error does not bound a different but correct float32 design. About 0.8 % of
    rows are like this (0.72 % to 1.25 % in the emulated draws below). They are in-domain and occur
    in production, where `beta = sigmoid(b)` rounds to exactly 0.5 for `|b|` below about 0.004. A
    correct candidate must handle them, so the inputs are kept as they are. These rows are also the
    only place where the 16-ulp-shift control is visible: everywhere else 16 ulp is far below the
    baseline's bfloat16-level error, so an `sqnr` bound cannot work (the shifted control has the
    baseline's own SQNR, 55.6 dB) and neither can a fixed tolerance.
  - *Measured.* In the MI355X QA audit, under the comparator's earlier row rule (RMS over all 64
    columns), the float32 evaluation of the reference violated twice the baseline error on 6.6e-5
    to 9.0e-5 of the elements (peak up to 54x, bias z-score at most 1.2e-3). The CPU emulation
    below reproduces this under that rule (5.3e-5 to 1.1e-4 at 65 to 16384 tokens).
  - *CPU emulation, current comparator* (per-row RMS over the elements that are nonzero in the
    reference or the baseline, rows with fewer than 8 of them on the whole output's RMS, admitted
    violations within 16x the bound). The baseline was emulated three ways: bfloat16 `beta * k`,
    exact MFMA partial sums of 16 or 32 products accumulated in float32, and the decay as
    `exp2(fl((g_i - g_j) * log2 e))` (how Triton lowers `tl.exp` on AMD) or as an accurate `exp`.
    Fifteen correct float32 designs were compared against each: the float32 evaluation of the
    reference; bfloat16 or float32 operands with sequential, 4-, 8- or 16-term accumulation; `beta`
    applied after the Gram product; decay as `exp(g_i - g_j)`, through `exp2` (also with a 1-ulp
    hardware `exp2`), split as `exp(g_i) * exp(-g_j)` or `exp2(l_i) * exp2(-l_j)`, or from gates
    pre-scaled by `log2 e`; and the decay folded into float32 operands, `(beta k_i 2^(l_i)) .
    (k_j 2^(-l_j))`, with a sequential FMA dot or 4-term MFMA groups. Every performance case (4096
    to 409600 tokens) was run with two draws. Violations occur only in the power-of-two-`beta`
    rows (whole outputs up to 4096 tokens: peak at most 0.5x elsewhere, at `k = 2`), so the larger
    cases were evaluated on those rows with the limits computed from the whole output.
    - Correct designs, performance cases, `k = 8`: violation fraction at most 1.25e-4 (operands
      with folded decay, FMA dot) and at most 5.0e-5 for every other design; the float32 reference
      evaluation 2.7e-6. Peak at most 7.5x for fourteen designs; 8.8x to 10.3x for the folded-decay
      FMA design.
    - Correctness cases: at most 2.9e-4 over 60 draws of 65 tokens and 1.5e-4 over 30 draws of
      1000 tokens (three baseline models each).
    - Controls: the 16-ulp shift violates on 1.0e-3 to 1.2e-3 in every performance case (3.5e-4 to
      1.9e-3 at 65 tokens); fp8 rounding of the output on about 0.36 of the elements, with peaks
      above 1e6x.
  - *Choice.* `k = 8`, the largest allowed value. At the peak elements the limit is dominated by
    the output rounding, so the peak falls only slowly with `k` (at `k = 2`: 14.8x for the float32
    reference evaluation, and 24x to 33x, past the 16x limit, for the folded-decay FMA design).
    `max_violation_fraction: 4.0e-4` is 3.2x above the worst correct design in the performance
    cases, 1.4x above the worst 65-token draw, and 2.5x below the lowest violation fraction of the
    16-ulp-shift control in a performance case.
  - *Limitation.* The folded-decay FMA design reaches 8.8x to 10.3x the bound at isolated elements:
    late rows of a chunk (`g` near -6), adjacent tokens and large `|A|`. There, its two factorized
    exponentials carry about 10 to 20 ulp of relative error, while the baseline's error is about
    0.1 to 0.2 ulp. This is inside the comparator's 16x limit, but with less than a 2x margin, and
    no allowed `k` removes it. The bound is kept as is: the decay-folded sequential-FMA design peaks
    at 8.8x to 10.3x, under the 16x limit, so every emulated correct design passes while the 16-ulp
    shift control is rejected in every performance case.
