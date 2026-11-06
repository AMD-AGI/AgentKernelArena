# write_req_to_token_pool, int32 table of 2048 x 202756

Scheduler bookkeeping of an SGLang prefill step: for each request the cached prefix slot indices and
the newly allocated slot indices of its extension are written into that request's row of the
request-to-token table (`req_to_token`, int32). The table is a mutable input; the evaluator restores
its initial (random) contents before every call and checks the whole table, so writes beyond a
request's `seq_lens`, to another row, or a wrong extension offset are detected. All values are
integers, so the table is compared `exact` against the integer reference; a tolerance would be
meaningless. This departure from the default bound is intentional.

- Production origin: SGLang `write_req_to_token_pool_triton`
  (`python/sglang/kernels/ops/memory/common.py`), called from `write_cache_indices` in
  `srt/mem_cache/allocation.py` for every prefill batch. The table shape 2048 x 202756 (requests x
  context length) is the configuration of the task this one replaces. Each request has a prefix of
  0 to 16 cached tokens and an extension of `max_extend_len` minus 0 to 8 tokens; the `max_extend_len`
  axis sets the extension size (decode-like 1 up to 4096-token chunks).
- Baseline: the SGLang Triton kernel (one program per request, 512-element tiles) at the recorded
  revision. SGLang hands the kernel one device pointer per request prefix, which a task input cannot
  express; the prefixes are therefore the rows of one `[n_requests, 16]` int64 tensor and the kernel
  offsets by the row stride (the only change to the kernel). The kernel recomputes each request's
  offset into `out_cache_loc` with a loop over all earlier requests, which makes large batches
  latency-bound; that is part of the baseline.
- Workloads: `n_requests` and `max_extend_len` both vary, and the size of a case is their product.
  Small batches run below the 10 us timing floor and are correctness-only. Timing risk: the
  (512, 16) and (256, 1008) performance cases rest on the estimated latency of the prefix-sum loop
  and the tile loops (the data moved is tiny) and may fall under 10 us on MI355X; drop or enlarge
  them if the GPU audit shows that. The largest cases (up to 2048 x 4096 tokens per step) exceed a
  typical chunked-prefill budget and exist to give the 100x span.
