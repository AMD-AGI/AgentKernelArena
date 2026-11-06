# bincount_penalty_state, vocabulary 151936

Builds the sampling-penalty state of a batch of requests from their token histories, as vLLM's
v1 GPU worker does when requests with repetition/frequency/presence penalties are added. For each
request slot it writes

- `prompt_bin_mask[slot]`: a bitmask over the vocabulary (4748 int32 words) with the bit of every
  distinct prompt token set (token `t` sets bit `t % 32` of word `t // 32`; bit 31 is the sign bit),
- `output_bin_counts[slot]`: the histogram over the vocabulary of the tokens from `prompt_len` up to
  `prefill_len` (the output tokens of a request resumed after preemption).

`idx_mapping` maps the batch to state slots; here it is a permutation of all `n_reqs` slots, so both
outputs are written completely (the evaluator poisons them first, so every row, including the zero
entries, must be written). Token ids at or beyond `prefill_len` hold valid tokens and must not be
counted.

- Production origin: `bincount` / `_bincount_kernel` in `vllm/v1/worker/gpu/sample/penalties.py`
  (`PenaltiesState.apply_staged_writes`). The vocabulary 151936 is the Qwen tokenizer size. Prompt
  and prefill lengths up to 32768 cover long-context prompts.
- Comparison: both outputs are integers (a bitmask and counts) and are compared `exact`; a tolerance
  would be meaningless. This departure from the default bound is intentional.
- Baseline: the vLLM Triton kernel, unchanged, preceded by a small Triton kernel that zeroes the
  mapped rows. vLLM does that with two advanced-index assignments (`prompt_bin_mask[idx_mapping] = 0`)
  before the launch, which `run()` may not do; the zeroing kernel is therefore part of the baseline
  (and of every candidate's work). vLLM also passes `max_prefill_len` from the host; here it is the
  column count of `all_token_ids`.
- Workloads: `n_reqs` and `max_prefill_len` both vary; case size is their product. Each row's
  histogram is 0.6 MB, so the zeroing and the cold histogram rows dominate: batches below 128
  requests (and the boundary cases) are correctness-only. Prefill lengths are up to a quarter shorter
  than `max_prefill_len`, half of the slots have output tokens, and about a third of the tokens come
  from a small set of frequent tokens so that atomics contend.
- Timing risk: the (128, 512) performance case rests on the estimated ~80 MB of histogram writes and
  the atomic traffic; it may fall under 10 us on MI355X. Drop or enlarge it if the GPU audit shows that.
