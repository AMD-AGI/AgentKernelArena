# apply_staged_writes, int32 buffer of 1024 x 131072

vLLM's v1 model runner keeps per-request state (block tables, token ids, ...) in persistent GPU buffers.
Each scheduler step it stages the changed segments on the host and applies them with one kernel launch,
one program per write: write `i` copies `write_contents[cu_lens[i-1]:cu_lens[i]]` into row
`write_indices[i]` starting at column `write_starts[i]`.

- The buffer is a mutable input; the evaluator restores its (random) prior contents before every call and
  the whole buffer is checked, so a write to the wrong row, offset or length is detected. All values are
  integers, so `exact` comparison is the only meaningful bound (an intentional departure from the default).
- Rows of different writes are distinct (concurrent writes to one row would race). Segment lengths of a
  batch average `seg_len`, range from 0 to `2 * seg_len` and sum to `n_writes * seg_len`; start offsets are
  random but every segment fits its row.
- Baseline: vLLM's `_apply_write_kernel` (`vllm/v1/worker/gpu/buffer_utils.py`), single-buffer form (the
  checkout also has a multi-group variant with a pointer table, which is not part of this scenario),
  `BLOCK_SIZE=1024`, one program per write looping over its segment.
- Workloads: production block-table appends and token-id writes are a few elements to a few thousand per
  request and launch-bound (a few microseconds); those are correctness-only. The timed cases are
  writes of long prompts (16384 and 65536 mean tokens per write, up to a full batch of 1024 writes) into
  a 131072-token row; the largest are beyond a typical step budget and give the 100x span. Timing risk:
  the baseline's per-write loop makes the smallest timed cases (8 writes of ~16K tokens) latency-bound;
  their time is an estimate (about 16 sequential 1024-element iterations per program) and may fall below
  10 us on MI355X. Drop or enlarge them if the GPU audit shows it.
