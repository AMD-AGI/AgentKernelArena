# Mamba2 SSD varlen chunk state, bfloat16, 128 heads x 64 channels, state 128, chunk 256

Computes, for every sequence of a packed prefill batch, the SSM state at the end of the sequence: the
decayed sum of the outer products `x_k (x) B_k` of the sequence's tokens in its last chunk, plus the carried
state (the state entering that chunk when the sequence began earlier, or the sequence's cached initial state
when it begins inside the chunk, decayed from the start boundary). It is the final stage that extracts the
per-sequence recurrent state to cache after a chunked Mamba2 prefill.

- Production origin: `chunk_state_varlen` in SGLang's Mamba2 SSD ops (adapted from vLLM). The baseline is its
  `_chunk_state_varlen_kernel` with the production default 16 x 16 x 16 tiles, one program per
  (tile, sequence, head), initial states always present. Current vLLM does not carry a separate varlen
  kernel (it reads the last chunk's state from the state passing output), so SGLang is the source.
- Scenario: 128 heads of 64 channels, 8 groups (16 heads per group), state size 128, chunk 256, float32
  chunk and initial states and outputs. Variable axes: `n_tokens` and `n_seqs`. The generator packs the
  sequences back to back at uniformly random cut points (each at least one token), so sequences begin and end
  anywhere inside a chunk; chunk boundaries ignore sequence boundaries as in the production layout.
- Workloads: packed prefill batches from 2048 tokens / 2 sequences to 131072 tokens / 128 sequences. The
  single-sequence, one-chunk and tiny batches are correctness-only: their kernel time is dominated by launch
  latency. Every sequence reads one chunk of `x`, `B`, `chunk_states` and `initial_states` (about 16 MB), so
  time scales with `n_seqs`. The correctness-only `t128_s128` case packs 128 one-token sequences (requests whose
  prompt hits the prefix cache except for its last token, at the `n_seqs` maximum); see the bound below.
- Inputs: `dt` is U(0, 0.1) for every slot. The padding slots of a partial last chunk (token >= `n_tokens`) are
  never read, because every sequence ends at or before `n_tokens` and the operator only sums its own tokens.
  They keep their draw instead of the chunk cumsum stage's zeros, which made the one-token case 255/256 exact
  zeros (the QA input audit rejects that as a degenerate draw). `dA_cumsum` still masks them (constant in the
  padding), so every value the operator reads is bitwise the same as with zero padding.
- Reference: float64, independent vectorised evaluation of the formula in the task description (validated
  against a per-token loop on small shapes). The kernel rounds `B * scale` to bfloat16 before the dot, a
  rounding the baseline-relative bound accounts for.
- Bound: `baseline_relative` with the default `k: 2` and `max_violation_fraction: 0.001`. The baseline is not
  more accurate than float32 accumulation in general, but in rows where a sequence contributes one or a few
  tokens with a tiny channel value `x` (or a tiny `dt`), the outer-product term lies below float32 resolution
  of the carried state. There the baseline's own error is a single float32 rounding of
  `acc + S * exp(dA[L-1] - dA_b)` (one FMA), and the bound shrinks to one or two float32 ulps. A correct float32
  candidate that rounds that combination differently (separate multiply and add, an accumulator initialised
  with the decayed state, an `exp` that differs by one ulp), or whose float32 scale before the bfloat16 rounding
  of `B * scale` differs by one ulp, lands just outside twice the baseline's error on a few such elements. The
  MI355X QA audit's float32 PyTorch evaluation of the reference failed the default bound in t131072_s128 on one
  element (violation fraction 7.5e-9, peak 1.25x, bias z 2e-4) and reached 0.73x to 0.93x in t1_s1,
  t16384_s16 and t32768_s32. A CPU emulation of the kernel's arithmetic reproduces the audit's baseline SQNR
  within 1.3 dB on the small cases. It models `tl.exp` lowered to `exp2(x * log2e)`, `B * scale` rounded to
  bfloat16, float32 16-token dots and an FMA carry. In that emulation, such candidates reach violation fractions
  up to 2.8e-4 over 200 seeds of t1_s1 (median at most 6e-6) at peaks up to 1.5x the bound, up to 1.2e-5 at
  peaks up to 2.1x in `t128_s128`, and up to 1.6e-6 at peaks up to 2.3x in the large cases. Those peaks are far
  under the 16x admission limit, and a fraction of 0.001 admits them. Raising `k` alone would not help, because the same candidates still exceed `k = 4` on a few elements of
  t1_s1. In the same emulation the audit's 16-ulp shift control violates 2.1e-3 to 1.8e-2 of t1_s1 and 6.2e-3
  to 7.0e-3 of `t128_s128` (peaks 12x to 14x), so it stays rejected; `t128_s128` keeps that margin stable
  because its 128 one-token sequences average out the few float32-limited rows of a single sequence. Fp8 rounding and zeroed, half-written or stale outputs
  violate orders of magnitude more elements.
- Baseline adaptation to Triton 3.8.0 (gfx950): upstream picks the carried-state pointer (`chunk_states`
  when the sequence began before its last chunk, otherwise `initial_states`) inside `if` branches and loads
  it once after them, so the branches yield a pointer tensor. Triton's AMD backend gives every pointer
  argument whose buffer is at most 2 GiB a 32-bit range for buffer ops. In the largest case (131072 tokens,
  512 chunks) `chunk_states` is exactly 2 GiB and `initial_states` is not, and the
  `TritonAMDGPUCanonicalizePointers` pass rewrites the two yielded pointers differently and fails
  (`scf.if ... region branch point has 3 operands, but region successor needs 2 inputs`). The vendored
  kernel performs the same masked load inside each branch and yields the loaded tile instead. The
  arithmetic, masks and memory accesses are unchanged: in the Triton interpreter on reduced shapes the
  adapted kernel's output is bitwise identical to the upstream kernel's, and a gfx950 compile with
  Triton's own argument specialization succeeds for every workload case (upstream fails only the largest).
- Timing: the baseline's small tiles make it compute-latency bound rather than bandwidth bound, so its
  times are well above the bandwidth estimate. The MI355X QA audit measured 52 us (t2048_s2) to 1.96 ms
  (t131072_s128).
- Coverage versus the schema-v2 task: that task used float16 inputs, uniform sequences and no initial states.
  This scenario uses bfloat16 (the production Mamba2 dtype) and always passes initial states, which exercises a
  superset of the kernel (both the chunk-state and the initial-state carry branches, mid-chunk starts and
  ends). The no-initial-states launch (`HAS_INITSTATES=False`) is not a separate scenario.
