# prepare_prefill_inputs, 1024 request slots, row length 131072

In vLLM's v1 model runner each persistent request slot holds all of its token ids
(`all_token_ids[max_num_reqs, max_model_len]`). For a prefilling request the step feeds the next chunk of
`query_len` of them to the model; this operator copies the chunk into the flat `input_ids` buffer and
stores the next `num_lookahead` prompt tokens of the slot (0 beyond the prompt) in
`next_prefill_tokens[k, slot]`, which speculative decoding uses. One program per request loops over its
chunk in blocks of 1024.

- `input_ids` is fully written and `next_prefill_tokens` is a mutable input (the evaluator restores its
  random prior contents before each call and checks all of it, so unscheduled slots must survive). Both
  are integers, so `exact` comparison is the only meaningful bound (an intentional departure from the
  default).
- Scenario: every scheduled request is prefilling (`num_computed_tokens < prefill_len`). The checkout's
  kernel also returns early for decode requests, leaving their `input_ids` slots to another kernel; such
  slots could not be an evaluated output, so the scenario leaves decode requests out. The baseline
  keeps the early return unchanged.
- Chunk generation: slots are distinct and random; chunk lengths average `query_len`, range from 1 to
  twice that (bounded by the row) and sum to `num_reqs * query_len`; each chunk starts at a random
  computed-token count and finishes the prompt exactly, leaves one or two tokens, or a random remainder.
  `num_lookahead` is 3, as with three speculative steps.
- Baseline: vLLM's `_prepare_prefill_inputs_kernel` (`vllm/v1/worker/gpu/input_batch.py`) unchanged,
  `BLOCK_SIZE=1024`.
- Workloads: decode-sized batches and short chunks are launch-bound (a few microseconds) and
  correctness-only. The timed cases are long prefill chunks (16384 and 65536 mean tokens per request, up
  to 1024 requests, the largest beyond a typical step budget). The `all_token_ids` table is 512 MB.
  Timing risk: the smallest timed cases (one request of 16K-64K tokens) run a serial load-store loop of
  16-64 iterations per program and are estimated at about 10-40 us on MI355X; drop any the GPU audit
  finds below 10 us.
