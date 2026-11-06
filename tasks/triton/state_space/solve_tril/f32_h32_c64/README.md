# solve_tril, float32, 32 heads, chunk 64

The block inverse of Gated DeltaNet chunked prefill. Per chunk of 64 tokens and value head, the
chunk's 64x64 block of `A` defines `L`, its strictly lower triangle (the output of the chunk
Gram-matrix step), and the result is `(I + L)^-1`, again lower triangular with a unit diagonal, in
the same `[token, head, column]` float32 layout. The upper triangle of the result, and its columns
beyond a partial last chunk, are zero. The diagonal of `A` and everything above it are not part of
the matrix and are ignored.

- **Origin.** `solve_tril` of flash-linear-attention as carried in vLLM
  (`vllm/third_party/flash_linear_attention/ops/solve_tril.py`), chunk size 64, with Qwen3-Next's 32
  value heads. It replaces three earlier tasks that benchmarked the stages of this routine
  (`solve_tril_16x16`, `merge_16x16_to_32x32`, `merge_16x16_to_64x64`); the chunk-32 path is not a
  production scenario for chunk size 64 and is dropped. The baseline is the 64x64 path, which inverts
  the four diagonal 16x16 blocks by forward substitution and merges them with 16x16 products.
- **Baseline changes.** Single sequence only (no variable-length or TMA path), the autotune replaced by
  one fixed launch (4 warps, 2 stages, `ieee` dot precision; not tuned on this hardware), six extra
  stores that write zeros into the strictly-upper 16x16 blocks (the production wrapper obtains those
  zeros from `torch.zeros_like`; here every output element must be written by a kernel), and the
  forward-substitution row loads masked to the strictly lower triangle with
  `tl.where(o_i < i, b_a, 0.)`, the mask FLA's newer fused chunk solve applies to the same rows
  (SGLang `python/sglang/kernels/ops/attention/fla/chunk_fwd.py`). The diagonal-block loads were
  already masked; the off-diagonal 16x16 blocks lie entirely below the diagonal. In the Triton
  interpreter (1 to 200 tokens) the masked kernel returns bit-identical outputs for noise or zeros
  above the diagonal, and the same bits as the unmasked kernel on zeros; the unmasked kernel fed
  noise falls to 17-18 dB SQNR.
- **Inputs.** `A ~ N(0, 0.05^2)` in every entry. The scale of the strictly lower triangle matches
  `beta * <k_i, k_j>` of unit-norm keys, so `I + L` is well conditioned (the inverse's off-diagonal
  entries stay small). Production feeds exact zeros on and above the diagonal; this task draws noise
  of the same law there instead, so that no input value is structurally constant or exact in a lower
  precision (a 1-token case would otherwise be all zeros, and about 51% of a full chunk is on or above
  the diagonal) and so that an implementation that reads those entries instead of ignoring them fails.
- **Workloads.** 4096 to 409600 tokens (100x span). Traffic is 16 KB per token (8 KB read, 8 KB written)
  so the smallest case is about 67 MB, an estimated 19 us at the bandwidth roofline, above the 10 us
  floor; the serial forward substitution makes the baseline slower still (57 us to 3.8 ms measured on
  MI355X). Token counts 1, 65, 100 and 1000 (partial chunks) are correctness-only. The largest case
  has about 3.4 GB of input and 6.7 GB of float64 reference output; the CPU pre-check skips it.
- **Comparison.** Reference float64 `solve_triangular` per chunk. `out` uses `sqnr` with
  `min_db: 135`, an override of the default `baseline_relative` (justified below).

## Bound override: `sqnr`, 135 dB

The default per-element bound rejects correct float32 implementations. On MI355X a float32
evaluation of the reference (`solve_triangular` in float32) failed it in every multi-token case at
1.25-1.30% of the elements (peak 3.6x-12.8x the bound). The per-row RMS term runs over the 64 columns
of a row, of which only `r` are nonzero in chunk row `r` (the diagonal is exactly 1 and the upper
triangle exactly 0 in every implementation), so rows near the top of each chunk get a limit diluted
by up to 8x, while two correct float32 algorithms make independent rounding errors there.

A CPU float32 emulation against the float64 reference (the baseline emulated with its 16x16
forward substitution and merges; SQNR over 65, 100 and 1000 tokens with three seeds and 4096 tokens
with two, out-of-bound fractions of the default bound against the emulated baseline at 4096 tokens)
gives:

| Implementation | default bound, out-of-bound elements | SQNR |
| --- | --- | --- |
| baseline algorithm (emulated) | (itself) | 154.3-155.3 dB |
| baseline, sequential row sums | 1.4e-4 | 154.3 dB |
| baseline, merges without FMA | 9e-4 | 154.2 dB |
| baseline, other association of the merge products | 6.3e-3 | 154.3 dB |
| `solve_triangular` float32 | 8.1e-3 | 154.0-155.1 dB |
| 64-row forward substitution | 7e-4 | 160.1-161.2 dB |
| recursive 2x2 block inverse | 1.1e-2 | 152.5-153.7 dB |
| doubling product `(I - L)(I + L^2)...(I + L^32)` | 1.4e-3 | 157.8-158.9 dB |
| Neumann series by Horner (63 products) | 3.1e-2 | 149.8-151.1 dB |
| tf32 merge products (wrong precision) | 37% | 82.8 dB |
| baseline x (1 + 16 ulp) (QA control) | 46% | 120.4 dB |
| bfloat16-rounded output | 49% | 66.8 dB |
| fp8-rounded output (QA control) | 49% | 42.6 dB |

Even `k: 8` leaves 1.0e-4 to 1.7e-4 of the elements of correct designs (`solve_triangular`, the
recursive inverse) out of bound, with a peak that grows with the case size, so no per-element bound
without a violation fraction accepts them, and a violation fraction would let that share of elements
be arbitrarily wrong. A whole-output SQNR bounds the total error energy instead. The baseline
measures 151.1-152.4 dB on MI355X (about 3 dB below its CPU emulation) and the least accurate correct
design 149.8 dB on the CPU; 135 dB sits 16.1 dB and 14.8 dB below them, 14.6 dB above the 16-ulp
shift control (20 log10 2^20 = 120.4 dB), and rejects reduced-precision products (tf32, bfloat16,
fp8) by 50 dB or more. The 1-token case is an exact identity in every implementation and rejects the
shift control on its own.
