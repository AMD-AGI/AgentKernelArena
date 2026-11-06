# Allowed-token, logit-bias and minimum-length masking, float32, vocabulary 151936

Per-request sampling constraints that act on single logits before sampling: an allowlist (every
token outside it is forbidden), additive per-token logit biases (the `logit_bias` request field), and
`min_tokens` handling (the stop/EOS tokens are forbidden until the request has produced enough
tokens). The three stages are applied in that order, in place, per logit row. An allowlist row writes
the whole vocabulary row (about 0.6 MB); bias and stop rows write a few scattered entries.

Production origin: vLLM v1 GPU model runner, `vllm/v1/worker/gpu/sample/logit_bias.py`
(`_bias_kernel`, launched by `apply_logit_bias`).

## Scenario and workloads

Float32 logits of the Qwen vocabulary, four logit rows per request (one verified plus three draft
tokens, at consecutive positions), request slots randomly permuted, table capacities as upstream
(1024 allowed ids, 1024 bias entries, 128 stop ids). The var axis is the number of logit rows.
Performance cases run from 128 to 12800 rows, a 100x span; smaller batches are correctness-only.
Because an allowlist row costs a full-row write, the workload is a batch where about half of the
requests use an allowlist (a constrained-decoding batch); with fewer allowlist requests the 10 us
floor would move to larger batches. This must be confirmed on the GPU.

Input generation (`inputs.py`): half of the requests have an allowlist of 2 to 1024 ids
(log-uniform), half have 1 to 128 biased ids, 30% have 1 to 16 stop ids with a minimum length that
is active for some of their rows (`min_lens = start position + [-6, 8]`, rows at four consecutive
positions). Bias and stop ids often reuse the request's allowed list, so the stage order is
observable (bias on a forbidden token stays `-inf`; stop ids override biases). Ids inside one list
are distinct, as the upstream tables come from sets.
Bias values are drawn for the whole 1024-wide table, including the unread tail past `num_logit_bias`
(stale entries of a persistent table in production); zero padding would make the table almost entirely
exactly representable values. Only the first `num_logit_bias[s]` entries are read by the reference and
the kernel.

## Correctness

The reference loops over rows in PyTorch following the three stages. The bias is a single
float32 addition, identical to the kernel's, and all other writes are `-inf` stores, so the whole
logits tensor is compared bit for bit (`exact`) against a float32 reference; this also checks that
every other logit is untouched. A float64 reference would only add memory (the logits tensor of the
largest case is 7.8 GB).

## Baseline

vLLM's `_bias_kernel` and launcher. Two additional `tl.debug_barrier()` calls are the task's only
changes: upstream already barriers inside the allowlist stage, but a token id that appears in both the
allowed list and the bias list (or the bias and stop lists) is read, stored and re-stored by different
waves of the workgroup, so without a barrier the order of those accesses is undefined. The
generator deliberately produces such overlaps, as real requests do.
