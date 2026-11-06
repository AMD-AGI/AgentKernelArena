# count_nans, float32 logits, vocabulary 151936

Counts the NaN entries of each row of the logits and writes one int32 per row. vLLM's v1 GPU worker
uses this count as a metric on the logits of the sampled rows. Positive and negative infinities are
not NaN and must not be counted.

- Production origin: `get_num_nans` / `_num_nans_kernel` in `vllm/v1/worker/gpu/metrics/logits.py`.
  The vocabulary 151936 is the Qwen tokenizer size; vLLM passes float32 or lower-precision logits,
  the scenario fixes float32.
- Comparison: the output is an integer count and is compared `exact`; a tolerance would be
  meaningless. This departure from the default bound is intentional.
- Baseline: the vLLM Triton kernel, one program per row with an 8192-element block loop, unchanged
  except for the `run()` wrapper and the `libdevice` import path (vLLM imports it from
  `torch._inductor`).
- Inputs: logits are standard normal; row kinds are mostly clean rows, some with a handful of NaNs,
  some with 5% NaNs and a few entirely NaN rows; every row also has a few infinities.
- Workloads: `n_reqs` is the only var axis. Each row is 0.6 MB and the baseline reads it with a single
  program, so decode batches below 32 rows are correctness-only. Large row counts correspond to
  prompt-logprob or large-batch evaluation; 4096 rows (2.5 GB of logits) give the 100x span.
- Timing risk: the 32- and 64-row performance cases rest on the estimated latency of one program per
  row (about 19 sequential 32 KB blocks); they may fall under 10 us on MI355X. Drop or enlarge them
  if the GPU audit shows that.
- Sibling scenario: `../f16_v151936` covers float16 logits (the dtype of the previous task).
