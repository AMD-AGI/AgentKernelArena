# GDN chunk state pass, bfloat16, 16 key heads, 32 value heads, K = V = 128, chunk 64

For one sequence of `n_tokens` tokens and each of the 32 value heads (key head `h // 2`), the operator
carries a float32 state `S[V, K]` across chunks of 64 tokens (Gated DeltaNet prefill,
`chunk_gated_delta_rule_fwd_h`):

```text
h[chunk]     = S                                  # state at the start of the chunk, rounded to bfloat16
v_new[chunk] = u[chunk] - w[chunk] S^T            # delta correction, rounded to bfloat16
S            = exp(g_last) S + (v_new * exp(g_last - g))^T k[chunk]
```

The state starts from the sequence's slot of a float32 state pool (`initial_state`, 8 slots, slot given
by `initial_state_indices`) and the final state is written back into that slot, so the pool is a
mutable input: it is restored before every call and checked afterwards like an output (the other slots
must stay unchanged). `g` is the chunk-local cumulative log-decay, restarting every 64 tokens.

- **Origin.** The chunked gated-delta-rule state kernel of SGLang's FLA package
  (`python/sglang/kernels/ops/attention/fla/chunk_delta_h.py`, adapted from flash-linear-attention),
  the hottest kernel of Qwen3-Next style GDN prefill. Dimensions are Qwen3-Next's (16 key heads, 32
  value heads, head dimensions 128, chunk 64). The earlier task of the same operator also covered
  tensor-parallel splits (8/16 heads), a Hg = H case and K = 64; these are separate scenarios and are
  not part of this one.
- **Baseline.** The earlier task's Triton extraction of the kernel, written as the standard
  fixed-length, scalar-gate path (one program per 32-wide value tile and value head, tile BV 32,
  4 warps, 2 stages), wrapped in `run()` to write into preallocated `h` and `v_new`. The variable-length
  option and the host-side chunk bookkeeping are not used by this scenario. The scenario is a single
  sequence (batch 1) as in the earlier task.
- **Inputs.** Keys have element standard deviation 128^-1/2 (the model L2-normalizes them) and `w` has
  0.1, which keeps the delta-rule recurrence contractive; the earlier task drew keys with unit variance,
  which makes the state grow without bound beyond about 1000 tokens, so it could not be used for long
  sequences. The log-gates are `0.03 * logsigmoid(x)` per token, accumulated within each chunk, which
  gives the state a memory of a few chunks.
- **Reference.** A float64 chunk-by-chunk implementation of the equations above with no intermediate
  rounding. The kernel rounds the state to bfloat16 for the `w S^T` product and the gated delta to
  bfloat16 for the update (its documented behavior); this is part of the baseline's own error, which
  candidates are held to. A CPU emulation of the kernel has 51.6-55.6 dB SQNR against the reference
  on `h`, 53.7-55.6 dB on `v_new` and 55.0 dB on the state pool (65.7 dB at 1 token).
- **Workloads.** 512 to 65536 tokens (128x span). The kernel is latency bound (it walks the chunks
  sequentially in 128 programs), so even the 8-chunk case is expected above the 10 us floor on
  MI355X, but this is the one the GPU audit should confirm. Shorter prompts are correctness-only:
  1 token (axis minimum), 100 and 200 (partial chunk) and 1000 (unaligned, partial last chunk).
- **Comparison.** `h` and `v_new` use `baseline_relative` with `k: 4` and `max_violation_fraction`
  1e-3 and 1e-4. The state pool uses `sqnr` with `min_db: 47`. The largest case needs about 4.3 GB
  for the float64 `h` reference plus the comparison temporaries.
  - *Why not the default.* The state carries every bfloat16 rounding of the sequence. Two equally
    valid implementations make slightly different roundings, so their states drift apart along the
    sequence. In the emulation, the difference between the baseline and the same algorithm with
    another accumulation order grows from 0.1x to 0.5x of the baseline's own state error between
    512 and 8192 tokens, while both errors stay at the same level. Element by element, twice the
    baseline's error then rejects correct designs. On MI355X the QA audit's float32 evaluation of
    the reference failed the default bound on one pool element at 1 token (2.4e-7 of the pool, peak
    1.17x). At 1 token each pool row's baseline error is one bfloat16 rounding (of that row's delta)
    times `k`, so a row where that rounding is almost exact gets a near-zero bound.
  - *Emulation.* The baseline was emulated on the CPU: bfloat16 state for `w S^T`, exact 16-term
    partial sums accumulated in float32, bfloat16 gated delta for the update. It was compared with
    these correct designs, for 1 to 16384 tokens (40 draws at 1 token, 1 to 6 draws elsewhere):
    (a) the same algorithm with sequential or 32-term accumulation, `exp` through `exp2`, or the
    gate computed as `exp(g_last) * exp(-g)`; (b) float32 throughout, including the float32
    reference evaluation and a fused multiply-add update; (c) a float32 state for `w S^T`;
    (d) a float32 gated delta for the update; (e) gating the bfloat16-rounded `v_new` and rounding
    again, the least accurate (2.5 dB below the baseline on the pool).
    Under the default bound:
    - (a) fails `h` on 2.3e-4 of the elements and the pool on 1.35e-3 at 16384 tokens; both still
      grow with length.
    - (c) fails `h` on 3.9e-4 and the pool on 1.9e-3.
    - (d) fails the pool on up to 5.5e-3 at 1 token.
    - (e) fails `h` on 1.2e-2, `v_new` on 9.6e-4 and the pool on 1.4e-2.
  - *`h` and `v_new`.* Both are bfloat16, so one output rounding dominates their bound. With
    `k = 4` the worst correct design, (e), violates on at most 8.9e-5 of `h` and 3.6e-7 of
    `v_new`; every other design violates on at most 7.5e-8. The evaluator's controls violate on far
    more: the 16-ulp shift on at least 0.84 (`h`) and 0.88 (`v_new`), fp8 rounding on about 0.43
    and 0.50 (falling slowly with length). Shortcuts with an fp8 state in `w S^T` or an fp8 gated
    delta violate on at least 0.12 and 0.22. The fractions leave 11x (`h`) and 280x (`v_new`) above
    design (e) and more than two orders of magnitude below the controls.
  - *State pool.* The pool is float32, so its bound has no output-rounding margin: any per-element
    bound compares two decorrelated accumulation errors. Design (d) violates the pool on 2.7e-3
    even at `k = 4` (1 token), while a wrong final state for one whole head is only 3.9e-3 of the
    pool, so no `k` and violation fraction separate them.
    - *Correct designs.* Pool SQNR is at least 52.4 dB, from design (e). The baseline has 55.0 dB
      from 1000 tokens, 55.8 dB at 100 tokens and 65.7 dB at 1 token. A final state rounded to
      bfloat16 has 52.8 dB, about as accurate as design (e), and is accepted.
    - *Wrong designs.* fp8 rounding has 31.5 dB, the fp8 shortcuts at most 38.8 dB from 100 tokens
      on (at 1 token `v_new` rejects them). A skipped decay of the last chunk has at most 7 dB from
      100 tokens on. One head of the slot left unwritten has at most 27.7 dB, and one unwritten row
      at most 39.5 dB from 1000 tokens on.
    - *Choice.* 47 dB is 5.4 dB below design (e) and 7.5 dB above the closest wrong design.
    - *Pool controls.* The untouched slots count as signal: zeroing 100 elements of another slot
      fails at 1 token, and 300 at 1000 tokens. A 16-ulp shift of the pool is 1e-6 relative, far
      below the baseline's own error, so the `sqnr` bound accepts it. That control is still
      rejected by `h` and `v_new`.
