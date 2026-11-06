# Kimi Delta Attention chunked prefill, bf16, 12 heads, head dimension 128

The complete prefill path of Kimi Delta Attention (KDA), the gated delta-rule linear attention of
Kimi-K3's linear-attention layers, as vLLM runs it on ROCm (`chunk_kda_with_fused_gate` with
`lower_bound=-5.0`, `use_qk_l2norm_in_kernel=True`, `output_final_state=True`). For every sequence
and head, with the state `S[v, k]` starting at `initial_state`:

```text
q_t <- q_t / sqrt(|q_t|^2 + 1e-6) * 128^-1/2        k_t <- k_t / sqrt(|k_t|^2 + 1e-6)
g_t  = lower_bound * sigmoid(exp(A_log[h]) * (raw_g_t + dt_bias[h]))     per key channel
S    = S * exp(g_t)                                  (decay along the key axis)
S    = S + sigmoid(raw_beta_t) * (v_t - S k_t) k_t^T
o_t  = S q_t                                        final_state = S after the last token
```

`q`, `k`, `v`, `raw_g`: `[n_seqs, seq_len, 12, 128]` bfloat16; `raw_beta`: `[n_seqs, seq_len, 12]`
bfloat16 (before the sigmoid); `A_log`: `[12]` and `dt_bias`: `[12 * 128]` float32;
`initial_state` and `final_state`: `[n_seqs, 12, 128 (v), 128 (k)]` float32; `o` like `q`.

- **Origin.** Kimi-K3 serving on MI355X: the model's KDA layers (96 heads of size 128, 12 per rank at
  tensor parallel 8, `gate_lower_bound = -5.0`) call `chunk_kda_with_fused_gate` for prefill. This
  task replaces the earlier image-kernel task of the same operator. That task also timed the packed
  decode entry point (`fused_recurrent_kda_packed_decode`); decode is a different operator and is
  not part of this task.
- **Scenario.** A batch of `n_seqs` prefill sequences of equal length `seq_len`, laid out as batch
  rows. vLLM packs sequences into one row with `cu_seqlens`; for equal lengths the fixed-length
  batch layout is the same memory and the same kernels (their `IS_VARLEN = False` variant), and it
  needs no host-side chunk-index preparation, which is not graph-capturable.
- **Baseline.** vLLM's ROCm KDA copy (`vllm/models/kimi_k3/amd/ops/third_party/kda/chunk.py` and
  `chunk_intra.py`) with the shared flash-linear-attention ops it calls (`chunk_delta_h.py`,
  `l2norm.py`), at the vLLM commit in `task.yaml`. `run()` is the production call sequence: two
  L2-norm kernels, the fused gate and chunk-local cumulative sum, the intra-chunk `Aqk` and the
  inverted `(I + Akk)` system (safe-gate sub-chunk kernel plus the fused inter/solve kernel),
  the `w`/`u`/`kg` recompute, the chunk state recurrence (which also writes `final_state`) and the
  output kernel. The kernel bodies are unchanged. Each `@triton.autotune` is replaced by one fixed
  entry of its own configuration space (listed in the files): several autotune keys do not include
  the token count, so the tuned configuration would depend on which case ran first, and tuning
  would benchmark inside audited calls. The production code writes the output into `v`; here it
  is written into `o` and `v` stays read-only. Only the default fused L2-norm kernel is kept.
  The baseline may be slower than the autotuned production code.
- **Inputs.** `q`, `k` and `v` are normal with the mean and standard deviation of SiLU(N(0, 1))
  (they follow a short convolution and SiLU in the model), `raw_g` and `raw_beta` are N(0, 1).
  `A_log ~ U(0, 1.5)` and `dt_bias ~ U(-8, 0)` give per-channel decays from none (about a third of
  the gate values are above -1e-4 per token) to fast (about a tenth are below -1 per token),
  so the output depends on state carried across many 64-token chunks and on `initial_state`.
  Recomputing every chunk from the initial state alone changes `o` by about 65% (relative RMS) and
  dropping `initial_state` by about 8%, against a baseline error of about 0.5%. With the earlier
  task's gate parameters every channel decayed by about e^-2.5 per token, so the chunk-to-chunk
  recurrence was numerically invisible. `initial_state ~ N(0, 0.1^2)`.
- **Reference.** A float64 token-by-token recurrence (above), independent of the chunked algorithm.
  The baseline run in the Triton CPU interpreter (with bfloat16 dots and round-to-nearest casts)
  reached about 47 dB SQNR on `o` and 48-61 dB on `final_state` against it.
- **Comparison.** `baseline_relative` with `k = 6` on both outputs instead of the default 2. The
  pipeline stores bfloat16 intermediates (normalized q and k, `Aqk`, the WY inverse, `w`, `u`,
  `kg`, chunk states, `v_new`), so a change of summation order flips some of their roundings and
  the flips propagate through the state recurrence. In the CPU emulation, the baseline with another
  entry of its own autotune space (reduction block 32 instead of 64 in the intra-chunk solve and the
  output kernel) needs `k` = 2.7 / 2.6 / 3.1 on `o` and 2.8 / 3.5 / 3.7 on `final_state` at 700 /
  2048 / 6144 tokens (only 62% of the final-state elements stay bitwise equal at 6144 tokens), so
  the default bound would reject equally accurate implementations, increasingly with sequence
  length. A second emulation with other draws found `k` = 1.5 / 1.9 / 2.9 / 3.0 on `o` and
  2.3 / 2.3 / 4.5 / 3.7 on `final_state` for 2 x 300, 16 x 128, 1 x 4096 and 1 x 8192 tokens.
  Its largest need, 4.5, sets the margin: `k = 6` covers it. An implementation that keeps the
  intermediates in float32 is more accurate than the baseline and passes any `k`. With `k = 6`
  the evaluator's controls (16-ulp shift, fp8 rounding, half-written and zero outputs), a 0.5%
  scale error on either output, a missing initial state and a 2% change of `lower_bound` are all
  still rejected in the emulation. The scale and `lower_bound` errors are rejected even at `k = 8`.
- **Workloads.** Total tokens 320 to 32768 (102x span). The 1080- and 7211-token cases are the
  packed prefill sizes in a Kimi-K3 serving trace (the trace does not record the sequence split, so
  they are single sequences, as in the earlier task); the others are batches of 1024-token requests
  (the serving benchmark's input length), 32 requests of 512 tokens and long-context prefills. The chunk
  state recurrence is sequential over chunks, so long single sequences take milliseconds. Axis
  minimum (one token), five one-token sequences, exactly one chunk, partial chunks (100, 3 x 77)
  and two unaligned 1000-token prompts are correctness-only. The float64 reference loops over tokens
  (on the order of a second per 10k tokens on a GPU, about 8 s for 32768 tokens on 4 CPU threads).
