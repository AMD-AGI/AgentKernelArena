# Bad-words logit masking, float32, vocabulary 151936

Stops a model from emitting chosen multi-token phrases. Each request owns up to 128 "bad words"
(token sequences). At every logit row, the operator checks whether the tokens just before the
current position equal a bad word without its last token, and if so sets the logit of that last
token to `-inf`. The "tokens just before" are the request's generated tokens followed by the
speculative draft tokens already placed at earlier rows of the same request, so one request can span
several logit rows. Single-token bad words are masked unconditionally. The operator only writes
matched entries, so it is a sparse in-place update of the logits.

Production origin: vLLM v1 GPU model runner, `vllm/v1/worker/gpu/sample/bad_words.py`
(`_bad_words_kernel`, launched by `apply_bad_words`).

## Scenario and workloads

Float32 logits of the Qwen vocabulary (151936), four logit rows per request (one verified plus
three speculative tokens), request slots randomly permuted, up to 16 bad words per request (the
launch width `max_num_bad_words`). The var axis is the number of logit rows. Performance cases run
from 128 rows (32 requests) to 12800 rows (3200 requests), a 100x span; 8192 rows is also the size
of a prompt-logprob chunk. Smaller batches are correctness-only. The kernel launches
`n_tokens x 16` programs that each do a chain of dependent loads from cold caches, so the 10 us
floor is expected to be reached at roughly 128 rows; this must be confirmed on the GPU.

Input generation (`inputs.py`): bad words have 1 to 4 tokens, about 20% of slots have none, the
prefix tokens of the words and the generated history come from a pool of six token ids so that
prefixes match often (also across the generated/speculative boundary), and the final token of each
word is a random vocabulary id. Generated history lengths include 0 to 2 tokens so that the
"prefix longer than the history" early exit is exercised.

## Correctness

The reference (`reference.py`) vectorizes the semantics above in PyTorch over rows, one bad-word
slot at a time. The only change to the logits is `-inf` stores, so the whole logits tensor is
compared bit for bit (`exact`), which also proves that every unmatched logit is untouched. The
reference output is float32 (a copy of the input with `-inf` writes), as float64 would only add
memory.

## Baseline

vLLM's `_bad_words_kernel`, unchanged; `run()` replaces the launcher and passes the largest per-request
word count as the grid's second dimension (production reads it from the host-side request state).
