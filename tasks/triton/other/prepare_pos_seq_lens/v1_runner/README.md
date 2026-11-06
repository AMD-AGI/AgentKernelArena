# prepare_pos_seq_lens, 1024 request slots

vLLM's v1 model runner builds the per-step attention metadata on the GPU. This operator writes the
position of every scheduled token (`num_computed_tokens` of its request plus its offset in the chunk)
and the sequence length of every request, and zero-pads the unused `seq_lens` entries so a full CUDA
graph sees a fixed-size buffer. One program per request loops over its tokens in blocks of 1024; one
extra program does the padding.

- Outputs are integers (`pos` int64, `seq_lens` int32), so `exact` comparison is the only meaningful
  bound (an intentional departure from the default `baseline_relative`). `seq_lens` is poisoned before
  each call, so the zero padding must really be written.
- Request slots are distinct and random; query lengths of a batch average `query_len`, range from 1 to
  twice that and sum to `num_reqs * query_len`; `num_computed_tokens` is random per slot.
- Baseline: vLLM's `_prepare_pos_seq_lens_kernel` (`vllm/v1/worker/gpu/input_batch.py`) unchanged,
  `BLOCK_SIZE=1024`.
- Workloads: real metadata batches (a few hundred decode requests, or a few prefill chunks) take only a
  few microseconds and are launch-bound, so they are correctness-only. The timed cases are long prefill
  chunks (16384 and 65536 mean tokens per request, up to 1024 requests); the largest exceed a typical
  step budget and give the 100x span. Timing risk: this kernel only stores, so its per-program loop is
  throughput- rather than latency-bound. The single request of 16384 tokens (16 loop iterations) is
  estimated to fall under the 10 us floor and is correctness-only; the remaining timed cases (a single
  request of 65536 tokens and batches whose longest request exceeds 16K tokens) are estimated at 10 us or
  more but not measured. Drop any the GPU audit finds below 10 us.
