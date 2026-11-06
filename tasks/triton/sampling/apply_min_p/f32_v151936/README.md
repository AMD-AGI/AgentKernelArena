# apply_min_p, float32 logits, vocabulary 151936

In-place min-p filtering of the logits of each sampled row: with `m` the row maximum and `p` the
request's `min_p`, every logit below `m + log(p)` becomes `-inf`; rows whose request has `min_p == 0`
are untouched. Each row's request is looked up through `expanded_idx_mapping`.

**Production origin.** vLLM's v1 GPU model-runner sampler (`vllm/v1/worker/gpu/sample/min_p.py`),
applied to the `[num_tokens, vocab]` float32 logits before sampling. The baseline is that file's
Triton kernel and `apply_min_p`, one program per row with two passes over the row.

**Scenario.** Vocabulary 151936 (Qwen2/Qwen3 family), float32 logits; the number of rows is the only
varying axis. `expanded_idx_mapping` is a permutation, and `min_p` mixes disabled rows (`0`, one
row in 16), `1.0` (only the maximum survives, one row in 16) and values from 0.02 to 0.5.

**Workloads.** Decode batches of 1 to 1280 rows and prompt-logprob chunks of 2560 and 4096 rows. Rows
below about 40 take the baseline less than the 10 us timing floor (it is latency-bound, one program
per row) and are correctness-only; 40 to 4096 rows are the eight performance cases (a 102x span).

**Comparison.** `logits` compares `exact`: every element of the result is either the input value
or `-inf`, so any difference is a wrong mask, not rounding. The input generator moves every logit
lying within 1e-3 of a row's threshold clear of it (below), so the float32 kernel and the float64
reference cannot legitimately disagree about an element at the threshold; a candidate may compute the
threshold in either precision.
