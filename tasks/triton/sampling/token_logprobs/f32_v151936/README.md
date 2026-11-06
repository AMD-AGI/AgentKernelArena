# Token log-probabilities, float32, vocabulary 151936

For logprobs requests vLLM does not materialize the full log-softmax: a kernel computes the row maximum
and the log-sum-exp and emits the log-probability only at the requested token ids (the sampled token
followed by the top-20 candidates, 21 ids per row). The same kernel serves prompt logprobs, where every
prompt position of a prefill chunk is a row.

Production origin: vLLM v1 GPU model runner, `vllm/v1/worker/gpu/sample/logprob.py`
(`compute_token_logprobs`, `_topk_log_softmax_kernel`).

## Scenario and workloads

Float32 logits of the Qwen vocabulary, rows drawn from N(0, 2^2) (a realistic logit spread), 21 token ids per
row drawn uniformly, except that rows by index modulo 4 carry boundary ids: 0 and the last vocabulary
id (which lies in the partial last tile of the scan); the row maximum and minimum; a duplicated id. The var axis is the number of rows (1 to 3200). The kernel streams about 0.6 MB
per row twice; performance cases run from 32 rows (decode batches with logprobs) to 3200 rows (a
prompt-logprob chunk), a 100x span. Smaller batches are expected to fall below the 10 us timing floor
and are correctness-only (confirm on the GPU).

## Correctness

The reference is a float64 `logsumexp` gather. The output is float32 and uses `sqnr` with `min_db: 135`: the signal-to-noise ratio of the whole
`[n_rows, 21]` output against the float64 reference must be at least 135 dB.

Bound justification. The 21 outputs of a row share the error of that row's log-sum-exp, so they are not
independent samples. The default `baseline_relative` bound (per-element, with a bias test that treats every
element as independent) rejected a correct float32 implementation: a plain float32 PyTorch evaluation of the
reference (float32 `logsumexp` gather) failed `k=2` with isolated elements at up to 1.28x the bound, and
even with `k: 3` and `max_violation_fraction: 0.005` it failed the n32 case on the bias test alone (violation
fraction 0, peak 0.63x, bias z 0.37: with 32 rows there are only 32 independent errors, and z 0.37 is
unremarkable for 32 samples although it exceeds the 5/sqrt(672) threshold computed over 672 correlated
elements). The error is dominated by the row log-sum-exp, which float32 accumulation over 151936 terms
cannot make exactly rounded, and the baseline's tiled two-pass sum is only a little more accurate. An
SQNR bound measures the aggregate error and is not affected by that correlation.

Numbers (CPU emulation, 32 and 3200 rows, N(0, 2^2) logits, uniform ids): a float32 `logsumexp` gather
and a tiled two-pass float32 sum score 148.5 to 149.1 dB; the float32 rounding of the exact result alone
scores 152.5 dB. The known-wrong controls are far below: a shift by 16 float32 ulp scores 120.4 dB, zeros,
stale and unwritten rows and float8 rounding are tens of dB lower. 135 dB leaves about 13 dB (a 4.5x larger
error than float32 accumulation) for correct implementations and 15 dB above the 16-ulp shift. A skipped
vocabulary tile (1024 of 151936 logits) shifts the log-sum-exp by about 0.7% and scores about 65 dB. The
same threshold is used by `triton/normalization/log_softmax/f32_v151936`.

## Baseline

vLLM's `_topk_log_softmax_kernel` unchanged, launched as upstream does (one program per row, vocabulary
tiles of 1024, a two-pass max and exp-sum). The launcher writes into the preallocated output.
