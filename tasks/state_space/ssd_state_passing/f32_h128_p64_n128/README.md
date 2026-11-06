# Mamba2 SSD state passing, float32, 128 heads, flattened state 64 x 128, chunk 256

The inter-chunk recurrence of chunked Mamba2 prefill: within each sequence, the state after chunk `c` is the
state after chunk `c - 1` decayed by `exp(dA_cumsum[h, c, -1])` plus the chunk's own contribution, starting
from the sequence's cached initial state. The output holds the state after every chunk (4 MB per chunk:
128 heads x 64 x 128 float32), which the chunk scan stage then consumes.

- Production origin: `_state_passing_fwd_kernel` in vLLM's Mamba2 SSD ops. The baseline is that kernel at
  the pinned revision, which parallelises across (state tile, sequence, head) and loops over a sequence's
  chunks using `last_chunk_indices`. This replaces the older schema-v2 task's `seq_idx` formulation, which
  is no longer the production kernel: production chunks never cross a sequence, and the chunk range of each
  sequence follows from `last_chunk_indices`. The autotune is replaced by a fixed `BLOCK_SIZE` of 1024 so a
  first call never tunes inside a graph capture, and `fast_exp` (the hardware `exp2` instruction) is inlined.
- Scenario: 128 heads, state 64 x 128 flattened to 8192, chunk 256, float32 everywhere, initial states always
  present. Variable axes: `n_chunks` and `n_seqs`; the generator splits the chunks into consecutive non-empty
  groups at uniformly random cut points.
- Workloads: 8 to 512 chunks in 1 to 16 sequences for performance (each case moves 8 MB per chunk, 64 MB to
  4 GB); a single chunk and tiny or ragged batches are correctness-only, being below the 10 us floor.
- Reference: float64 sequential recurrence per sequence, independent of the kernel. `out` uses `sqnr` with
  `min_db: 135`, an override of the default `baseline_relative` bound (justified below).
- Timing: the chunk loop is serial within a sequence, so low `n_seqs` cases are latency bound; the MI355X
  QA audit found every performance case, including the smallest (8 chunks, 1 sequence), at or above the
  10 us floor.
- Coverage versus the schema-v2 task: that task used the obsolete `seq_idx` formulation and a smaller state.
  This scenario always passes initial states; the `HAS_INITSTATES=False` launch (zero start state) is not a
  separate scenario.

## Bound override: `sqnr`, 135 dB

The default per-element bound rejects correct float32 implementations. On MI355X a float32 evaluation of
the reference (a `torch.exp` decay, the multiply and the add rounded separately) failed it in all 13 cases,
at 0.019-0.18% of the elements with a peak of 1.5x-4.5x the bound. The baseline is not more accurate than
float32 accumulation; it is float32 accumulation itself (one fused multiply-add per chunk, a hardware
`exp2` decay). Every correct design rounds the state afresh at every chunk and carries the error on with the
decay (up to about 0.77 per chunk in the slowest heads), and the decay itself, one value per head and chunk,
differs by an ulp or two between exponential implementations, which shifts a whole row of 8192 elements
coherently. Elements in the tails of a row then exceed twice the baseline's per-row RMS error, even when the
total error is the same as the baseline's.

A CPU emulation against the float64 reference (8 of the 128 heads, inputs drawn by `inputs.py`; c1_s1,
c3_s2, c5_s3, c37_s37, c8_s1 and c16_s2 with two seeds, c32_s2, c64_s4, c100_s7, c128_s4, c256_s8, c512_s8
and c512_s16 with one; out-of-bound fractions and peaks of the default bound and of `k: 8`, both against the
emulated baseline) gives:

| Implementation | default bound, out-of-bound elements (peak) | `k: 8` peak | SQNR |
| --- | --- | --- | --- |
| baseline algorithm (FMA, correctly rounded `exp2`) | (itself) | (itself) | 149.5-155.0 dB |
| baseline algorithm, `exp2` off by up to 1 ulp | up to 3.2e-2 (5.7x) | 1.70x | 145.2-151.0 dB |
| baseline algorithm, `exp2` off by up to 2 ulp | up to 8.6e-2 (9.3x) | 2.50x | 140.5-149.7 dB |
| `exp` decay, separate multiply and add (the probe) | up to 2.4e-3 (3.8x) | 1.24x | 148.3-153.6 dB |
| `exp` decay, FMA | up to 3.7e-4 (2.4x) | 0.86x | 150.0-156.6 dB |
| `exp2` decay, separate multiply and add | up to 2.0e-3 (3.3x) | 1.05x | 148.1-152.2 dB |
| Hillis-Steele scan over the chunks of a sequence | up to 1.8e-2 (3.2x) | 1.60x | 144.2-152.7 dB |
| blocks of 8 chunks, then `S = P * S_start + local` | up to 1.9e-3 (3.2x) | 1.09x | 146.7-155.0 dB |
| control: output scaled by 1 + 16 u (`shifted_16_ulp`) | | | 120.4 dB |
| control: decay rounded to bfloat16 | | | 60.3-69.9 dB |
| control: decay read from `dA_cumsum[..., chunk_size - 2]` | | | 54.4-58.6 dB |
| control: output rounded to fp8 e4m3 | | | 31.5 dB |

Every correct design fails the default bound, and even the baseline's own algorithm with a different
`exp2` (one value per row) or a parallel scan fails `k: 8`, the largest allowed multiple, so no per-element
bound without a violation fraction accepts them, and a violation fraction would let that share of the
elements be arbitrarily wrong. A whole-output SQNR bounds the total error energy instead. 135 dB, a relative
RMS error of 1.8e-7 (about 3 float32 unit roundoffs), sits 14.5 dB below the emulated baseline, 9.2 dB below
the least accurate other design (the scan) and 5.5 dB below the most pessimistic model of the hardware
`exp2` (2 ulp), 14.6 dB above the 16-ulp shift control (20 log10 2^20 = 120.4 dB) and 65 dB or more above
a reduced-precision decay, an off-by-one decay index and fp8 outputs. A single element of RMS magnitude
wrong by that magnitude gives 87 dB even in the largest case (512 chunks, 537 M elements), so isolated
wrong elements stay rejected.
