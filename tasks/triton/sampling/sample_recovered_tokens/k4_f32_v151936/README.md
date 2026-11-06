# Recovered-token sampling, float32, 4 draft tokens, vocabulary 151936

When speculative decoding rejects a draft token, the rejection sampler resamples a replacement from
the residual distribution `max(target - draft, 0)` (unnormalized). vLLM draws it with the Gumbel-max
(exponential race) trick: one exponential noise row per request, shared by all of the request's draft
positions, and `argmax(prob / q)`; production passes the reciprocal `inv_q` so the kernel multiplies.

Production origin: vLLM v1 rejection sampler, `vllm/v1/sample/rejection_sampler.py`
(`sample_recovered_tokens` and `sample_recovered_tokens_kernel`), the path taken when the draft model
supplies probabilities.

## Scenario and workloads

Float32 probabilities of the Qwen vocabulary (151936), exactly 4 draft tokens per request (so
`cu_num_draft_tokens` is `4, 8, 12, ...`), the var axis being the number of requests (1 to 512, so 4 to
2048 draft positions). The kernel reads two probability rows per position plus one noise row per request,
about 1.8 MB per request; the performance cases run from 5 to 512 requests (a 100x span). Smaller
batches are expected to fall below the 10 us timing floor and are correctness-only (confirm on the GPU).
`draft_token_ids` is part of the production call but is not read on the draft-probability path; the
no-draft-probability path (`draft_probs is None`) is not part of this scenario.

## Inputs and correctness

Target probabilities are the softmax of logits drawn from N(0, 2^2); draft probabilities are the softmax of
the same logits plus N(0, 1) noise, so the residual is positive on about half the vocabulary and has the
cancellation structure of a real draft model. The result is an argmax, hence compared bit for bit
(`exact`). A float64 reference and the float32 kernel could disagree on the winner of a near-tie, so
the generator guarantees a relative gap of at least 1e-4 between the best and second-best score of every
position (it raises the target probability of the winning token by 0.03% where needed); the gap is far
above float32 rounding of the subtraction and product. The reference evaluates
`argmax(max(target - draft, 0) * inv_q)` in float64, per position, with the request of a position found
from `cu_num_draft_tokens`.

## Baseline

vLLM's `sample_recovered_tokens_kernel`, unchanged, launched as upstream does (grid of requests x
draft positions, vocabulary tiles of 8192, first-maximum tie rule). One program handles one draft
position, so small batches are latency-bound rather than bandwidth-bound. GPU timing at the smallest
performance case is the main risk for the 10 us floor.

## Coverage not carried over from the earlier task

Ragged and empty requests, and the path without draft probabilities (`draft_probs is None`), are not part
of this scenario: the interface has a fixed number of draft tokens per request (the usual case of
draft-model and MTP speculation) and always passes draft probabilities. A task for those variants needs
its own scenario.
