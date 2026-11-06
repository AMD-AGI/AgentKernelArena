# prepare_eagle_inputs, 1024 request slots

After the target model of an EAGLE speculative-decoding step, vLLM's v1 model runner prepares the draft
model's first input on the GPU: each request's target token ids are shifted by one position (the draft
model sees the next token), its last position is replaced by the token that continues the sequence (the
last sampled token, or the next prompt token for a chunked-prefill request), positions are copied, the
rejected tail of the request is dropped, and the per-request metadata (`last_token_indices`, start
offsets, sequence lengths) is filled and padded to `max_num_reqs` for CUDA graphs. One program per
request loops over its tokens in blocks of 1024; the last program does the padding.

- Outputs are integers, so `exact` comparison is the only meaningful bound (an intentional departure from
  the default). `draft_input_ids` and `draft_positions` are persistent buffers: they are mutable inputs
  that are zero before the step, so the slots of rejected tokens must stay zero and every other slot is
  checked. The four metadata outputs are fully written and poisoned before each call.
- Request slots are distinct and random; query lengths average `query_len`, range from 1 to twice that and
  sum to `num_reqs * query_len`; up to three trailing tokens (never all) of a request are rejected; about a
  quarter of the requests sampled nothing and take the next prefill token; positions continue from a
  random computed-token count.
- Baseline: the kernel behind `prepare_prefill_inputs` of vLLM's autoregressive speculator
  (`vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py`, named `_prepare_prefill_inputs_kernel`
  there), unchanged, `BLOCK_SIZE=1024`. It is the checkout's form of what earlier vLLM revisions called
  the EAGLE input preparation: the draft query start offsets, sequence lengths and step counter it also
  writes are part of the operator.
- Workloads: decode batches (a few draft tokens per request) are launch-bound and correctness-only. The
  timed cases are long prefill chunks (16384 and 65536 mean tokens per request, up to 1024 requests, the
  largest beyond a typical step budget). Timing risk: the smallest timed cases (one request of 16K tokens)
  rely on the per-program shift-and-copy loops (about 32 serial load-store iterations) and are estimated
  at about 10-30 us on MI355X; drop any the GPU audit finds below 10 us.
