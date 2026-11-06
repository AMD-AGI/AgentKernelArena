# Grammar-bitmask logit masking, float32, vocabulary 151936

Structured output (JSON schema, regex or grammar-constrained decoding) computes, for every
constrained request, a bitmask over the vocabulary: bit `v` of the packed int32 row says whether
token `v` is allowed at the next step. Before sampling, every disallowed logit is set to `-inf`,
in place. A mapping `logits_indices` tells which logits row each bitmask row constrains (a request
with speculative tokens owns several rows; unconstrained requests have none).

Production origin: vLLM v1 GPU model runner, `vllm/v1/worker/gpu/structured_outputs.py`
(`_apply_grammar_bitmask_kernel`, itself adapted from the xgrammar Triton kernel).

## Scenario and workloads

Float32 logits of the Qwen vocabulary (151936 = 4748 x 32, so no bit padding). Three quarters of the
rows are constrained (`n_masks = n_tokens - n_tokens // 4`), chosen at random, so the kernel must not
touch the other rows. Bitmask rows allow between 0.2% and 98% of the vocabulary (a per-row draw
from five densities) and use all 32 bits of every word, including the sign bit. The var axis is the
number of logit rows. Performance cases run from 128 to 12800 rows, a 100x span; the kernel moves
about a logits row of stores per constrained row, so 10 us is expected at roughly 100 rows
(confirm on the GPU). Smaller batches are correctness-only.

## Correctness

The reference unpacks the bitmask in PyTorch (chunked over rows) and writes `-inf` at the cleared
bits. As the only change to the logits is `-inf` stores, the whole tensor is compared bit for bit
(`exact`), which also checks that allowed logits and unconstrained rows are untouched. The reference
output is float32 (a copy of the input with `-inf` writes).

## Baseline

vLLM's `_apply_grammar_bitmask_kernel`, unchanged, launched as upstream does with block size 8192.
