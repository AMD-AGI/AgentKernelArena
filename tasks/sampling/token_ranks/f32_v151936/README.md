# Sampled-token rank, float32, vocabulary 151936

When a request asks for logprobs, vLLM reports the rank of the sampled token: the number of vocabulary
entries whose logit is greater than or equal to the sampled token's logit (so the best token has rank 1).
It is computed by a kernel that counts, per row, with one program per row.

Production origin: vLLM v1 GPU model runner, `vllm/v1/worker/gpu/sample/logprob.py` (`_ranks_kernel`, as
launched from `compute_topk_scores`).

## Scenario and workloads

Float32 logits of the Qwen vocabulary, rows drawn from N(0, 2^2). Rows by index modulo 8 are: 0, the
greedy token (rank 1); 1, the least likely token (rank equal to the vocabulary size); 2, a token whose
logit is duplicated at another position (checks the `>=` tie rule); 3 and 4, the first and the last
vocabulary id (the last lies in the partial tile of the scan); the rest, a uniformly drawn token.
The var axis is the number of rows (1 to 4096). The kernel streams about 0.6 MB per row once; performance
cases run from 40 rows (decode batches with logprobs) to 4096 rows (a prompt-logprob chunk), a 100x
span. Smaller batches are expected to fall below the 10 us timing floor and are correctness-only (confirm
on the GPU), and the smallest performance case is the main risk for that floor.

## Correctness

The output is an integer count, compared bit for bit (`exact`) with a float64 reference that counts
`logits >= logits[token]` (the float32 logits convert to float64 exactly, so ties are identical).

## Baseline

vLLM's `_ranks_kernel` unchanged, launched as upstream does (one program per row, vocabulary tiles of 8192).
The launcher writes into the preallocated output.
