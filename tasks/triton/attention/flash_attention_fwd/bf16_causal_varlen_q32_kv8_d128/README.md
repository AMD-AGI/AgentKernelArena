# Flash attention forward, causal varlen prefill, bfloat16, GQA 32/8, head_dim 128

Prefill attention over a packed batch of variable-length sequences with no KV cache
(`context_attention_fwd`): `q` is `[n_tokens, 32, 128]`, `k`/`v` are `[n_tokens, 8, 128]`, and
sequence `b` occupies tokens `b_start_loc[b] .. b_start_loc[b] + b_seq_len[b] - 1`. Each query attends
causally to the keys of its own sequence (positions at or before its own), query head `h` reading KV
head `h // 4`, with the scalar `sm_scale` (`1/sqrt(128)` unless a case sets it); the output is rounded to bfloat16. `max_input_len` is the
longest `b_seq_len` of the batch and sizes the launch grid, a host integer exactly as in production.

- **Operator origin.** The LightLLM/SGLang memory-efficient prefill kernel
  (`python/sglang/kernels/ops/attention/prefill_attention.py`); vLLM's
  `triton_prefill_attention.py` is an adaptation of the same kernel (exp2, sliding windows, sinks).
  In those projects it serves packed prompts without a paged cache: LLM prefill in the original
  LightLLM use, and encoder/vision attention (the non-causal mode) in SGLang and vLLM.
- **Baseline.** The SGLang Triton kernel from commit `887c401e`, `_fwd_kernel` unchanged: one program per
  (sequence, query head, 128-row query tile), online softmax in float32, probabilities rounded to
  bfloat16 before the second `tl.dot`, `BLOCK_M = BLOCK_N = 128` and 8 warps on gfx950. Only the
  `sglang.srt.utils` backend probes were inlined and the import-time `get_device_capability()` became a
  cached `get_device_properties(...).major` lookup; `run()` calls `context_attention_fwd` with
  `is_causal=True`.
- **Scenario.** bfloat16, causal, 32 query heads over 8 KV heads, head_dim 128 (Llama-3-8B and
  Mistral-7B class models). The earlier tasks mixed head counts (8-32), head_dim 64/128, MHA/MQA,
  float16 and a non-causal case; those are other scenarios of the same kernel. The vLLM variant's
  sliding windows are not part of this scenario.
- **Inputs.** `inputs.py` places one sequence of exactly `max_input_len` tokens at a random position
  and splits the remaining tokens over the other sequences with random lengths in
  `[1, max_input_len]`, so lengths and offsets change between the two correctness draws while the grid
  does not. q, k and v are standard normal; the reference is float64 and chunks the queries so the
  16k-token case stays within memory.
- **Workloads.** Single prompts of 1k-16k tokens and ragged batches of 4-64 prompts packed into
  8k-16k tokens. The shortest performance case (one 1024-token prompt, about 8.6 GFLOP of causal
  attention; its last query tile iterates 8 key tiles) is estimated at 15-20 us and must be confirmed
  above 10 us on MI355X. Batches of a few hundred tokens (including the source tasks' ragged cases
  `[7, 13]`, `[64, 200, 37]`, `[100, 100]` and `[333]`) are correctness-only. Two more
  correctness cases pack seven 1-token prompts with a 293-token prompt and use a non-default
  `sm_scale = 0.125` (query pre-scaling, as the vLLM source task's controls did), so a candidate
  must honour per-sequence offsets of single-token sequences and the scale argument.
- **Error bound.** `baseline_relative` with `k = 5` and `max_violation_fraction = 1e-6` (defaults 2
  and 0). The baseline's error is dominated by rounding the probabilities to bfloat16 before the second
  GEMM, and every other correct tiling of the online softmax (a different key-block size, unnormalized
  probabilities with a final division, exp2) produces independent noise of the same size. In a CPU
  emulation of the baseline arithmetic on one 3072-token sequence, such variants exceeded the bound at
  `k = 2` (2.4e-4 of the elements on 1024 tokens), `k = 3` and, by one element, `k = 4`; at `k = 5` their
  peak was 0.81-0.86x of the bound. The allowance (at most 67 elements of the 67M of the 16k-token cases,
  none below 1M) covers that tail without admitting one wrong query row (4096 elements). The same
  emulation rejects a causal off-by-one, a 1% scale error and probabilities rounded to FP8, and accepts
  float16 probabilities.
- **Not verified here.** Kernels, timing and the control suite need MI355X; only the CPU pre-check
  (contract, rendering, candidate rules, reference smoke) and a CPU emulation of the baseline
  arithmetic were run.
