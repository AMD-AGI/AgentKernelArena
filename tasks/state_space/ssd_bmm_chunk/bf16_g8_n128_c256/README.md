# SSD chunk Gram matrix C B^T, bfloat16, 8 groups, state 128, chunk 256

The `CB` term of Mamba2's chunked state-space-duality prefill. For every chunk of at most 256 tokens
and every group, `out = C_chunk @ B_chunk^T`, a `[256, 256]` float32 block (non-causal: all entries are
computed). `C` and `B` are `[n_tokens, 8, 128]` bfloat16; `cu_chunk_seqlens` holds the chunk start
offsets and ends with `n_tokens`, so the last chunk may be short. Rows and columns beyond a short
chunk's length are zero.

- **Origin.** `_bmm_chunk_fwd` of vLLM's Mamba2 SSD prefill
  (`vllm/model_executor/layers/mamba/ops/ssd_bmm.py`, from mamba-ssm), called as
  `_bmm_chunk_fwd(C, B, chunk_size, cu_chunk_seqlens, output_dtype=torch.float32)`. Dimensions are
  Nemotron-H's (8 groups, state size 128, chunk size 256). This replaces an earlier task with 16-bit float
  test data of 16 to 64 columns and chunks of 64 or 128.
- **Baseline.** vLLM's kernel unchanged; the autotune is replaced by one fixed configuration (128x128
  tile, K block 32, 4 warps, 2 stages), chosen from the production search list without tuning on this
  hardware, so a first call does not tune inside a graph capture. The dot operands are bfloat16 and the
  accumulator float32.
- **Inputs.** `C`, `B ~ N(0, 1)` in bfloat16; `inputs.py` builds the chunk boundaries of a single
  sequence (`0, 256, 512, ..., n_tokens`). Production batches also split chunks at sequence
  boundaries; that raggedness is limited here to the tail chunk.
- **Workloads.** 4096 to 409600 tokens (100x span). Each 256-token chunk writes 2 MB (256 x 256 x 4 B x 8 groups) and reads 0.5 MB, so the smallest case (16 chunks, 40 MB) is about 13 us at
  the bandwidth roofline. Token counts 1, 257 and 1000 (partial chunks) are correctness-only. The
  largest case writes 3.3 GB and its float64 reference is 6.7 GB; the CPU pre-check skips it.
- **Comparison.** The reference is float64 per-chunk matrix products. `out` uses `sqnr` with
  `min_db: 127`, an override of the default `baseline_relative` bound (justified below).

## Bound override: `sqnr`, 127 dB

The default per-element bound rejects correct float32 implementations. On MI355X a float32 PyTorch
evaluation of the reference (a float32 batched GEMM) failed it in all 8 cases: 1.13-1.18% of the
elements in every case of at least 1000 tokens (peak 3.6x to 5.7x the bound), 0.58% at 257 tokens (peak
8.4x) and 1 of the 8 nonzero elements at 1 token. The baseline is not more accurate than float32
accumulation in any way a candidate must match: it is float32 accumulation itself, with bfloat16 dot operands whose
products are exact in float32 and one accumulator rounding per MFMA step (16 or 32 products). A design
that rounds after every product (a scalar FMA loop), splits K, or uses another MFMA shape is equally
correct ("float32 or higher" accumulation) but rounds at other points, and the bound compares two
independent rounding errors element by element. In the 1-token chunks (case t1, and the tail chunk of
t257) a row holds one nonzero element, so where the baseline is exact the bound is one output rounding.

A CPU emulation against the float64 reference (exact bfloat16 products, float32 accumulator rounded
round-to-nearest-even, or truncated as a pessimistic model of the MFMA adder; 257, 1000, 4096, 16384 and
409600 tokens with two input draws each (the truncating models on 257, 1000 and 16384), the multi-chunk
cases subsampled to 4 chunks including the tail chunk; 1 token over 2000 draws) gives:

| Implementation | SQNR, 257 tokens and more | SQNR, 1 token (lowest / 1st percentile) | default bound vs baseline model (K block 16): out-of-bound elements (peak) |
| --- | --- | --- | --- |
| baseline model, one rounding per 16 products | 145.1 dB | 135.7 / 139.3 dB | (itself) |
| baseline model, one rounding per 32 products | 146.9-147.0 dB | 137.9 / 140.7 dB | 2.3e-4 (1.6x) |
| one rounding per 4 products (float32 MFMA) | 143.0 dB | 131.8 / 136.2 dB | 1.3e-2 (3.7x) |
| sequential FMA over K; CPU float32 matmul | 142.3 dB | 131.8 / 135.7 dB | 2.1e-2 (4.4x) |
| split K into 4 sequential parts | 144.8 dB | 130.6 / 138.4 dB | 4.8e-3 (3.3x) |
| pairwise tree over K | 145.5 dB | 136.9 / 139.3 dB | 2.6e-3 (2.7x) |
| truncating, one rounding per 16 / 32 products | 137.8-137.9 / 140.3 dB | 129.9 / 132.7 dB (16) | 9.7e-2 (5.1x) (16) |
| truncating, one rounding per 4 products | 134.7-134.8 dB | 125.9 / 129.2 dB | |
| control: output scaled by 1 + 16 u (`shifted_16_ulp`) | 120.4-121.4 dB | at most 123.1 dB | |
| control: output rounded to bfloat16 | 55.6 dB | 58.2-62.1 dB | |
| control: output rounded to fp8 e4m3 | 31.5-31.6 dB | 32.7-37.6 dB | |

Pairing the baseline model with the float32-MFMA model reproduces the MI355X probe (1.3% out of bound
at 4096 tokens against 1.17% measured). Even `k: 8`, the largest allowed multiple, leaves the sequential
FMA design out of bound on 8.7e-5 to 2.8e-4 of the elements, and the truncating models on up to 3.2e-3,
so no per-element bound without a violation fraction accepts them. A whole-output SQNR bounds the total
error energy instead. 127 dB, a relative RMS error of 4.5e-7 (about 7.5 float32 unit roundoffs), sits
15 dB or more below every round-to-nearest design and 7.7 dB or more below the truncating models in the
multi-chunk cases, 5.6 dB above the 16-ulp shift there (20 log10 2^20 = 120.4 dB) and 64 dB or more
above bfloat16 or fp8 outputs. The 1-token case (8 nonzero outputs) is the noisy one: over 2000 draws
every round-to-nearest design stays at least 3.6 dB above 127 dB (1st percentile 8.7 dB above), and
the 16-ulp shift stays 3.9 dB below it; only the truncating float32-MFMA model falls below it, in fewer
than 1% of the draws (lowest 125.9 dB). A single element wrong by the output's RMS magnitude gives 89 dB
even in the largest case (1600 chunks x 8 groups x 65536 = 839 M outputs), so isolated wrong elements,
a skipped K block or a misplaced chunk boundary stay rejected; unwritten elements are NaN and rejected
as non-finite.
