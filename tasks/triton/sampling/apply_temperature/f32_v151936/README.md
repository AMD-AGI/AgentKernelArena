# apply_temperature, float32 logits, vocabulary 151936

In-place temperature scaling: `logits[i, :] /= temperature[expanded_idx_mapping[i]]` in float32,
skipped for rows whose temperature is exactly 0 (greedy) or 1.

**Production origin.** vLLM's v1 GPU model-runner sampler
(`vllm/v1/worker/gpu/sample/gumbel.py`, `_temperature_kernel` and `apply_temperature`), applied
to the `[num_tokens, vocab]` float32 logits before sampling. The baseline is that kernel, a grid of
(row, 8192-wide column block).

**Scenario.** Vocabulary 151936 (Qwen2/Qwen3 family), float32 logits, normal(0, 2) values; the
number of rows is the only varying axis. `expanded_idx_mapping` is a permutation; temperatures are
uniform in [0.3, 1.5], except one row in 16 with temperature 0 and one in 16 with temperature 1 (the
early-exit paths).

**Workloads.** Decode batches of 1 to 1280 rows and prompt-logprob chunks of 2560 and 4096 rows. Fewer
than about 40 rows move too little data to exceed the 10 us timing floor and are correctness-only; 40
to 4096 rows are the eight performance cases (a 102x span).

**Comparison.** Default `baseline_relative` bound against a float64 division; the division is a single
rounding in float32 so a candidate within about one ulp of the baseline passes. Rows that must stay
unmodified (temperature 0 or 1) are checked against the unchanged input.
