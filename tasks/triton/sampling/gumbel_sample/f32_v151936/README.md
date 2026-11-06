# Gumbel-max sampling, float32 logits, vocabulary 151936

Samples one token per logits row with the Gumbel-max trick, as vLLM's v1 GPU model runner does after
all logits processors have run. Rows of requests with temperature 0 take the plain argmax; the others
take `argmax(logits + gumbel_noise)` where the noise comes from Triton's Philox-4x32-10 generator:
the row's stream key is `tl.randint(seed[state], pos[row])` and the uniform draw for vocabulary index
`v` is `tl.rand(key, v)`. The output is the int64 token id and is compared **exactly**.

Production origin: `vllm/v1/worker/gpu/sample/gumbel.py` (`gumbel_sample`). Production applies the
temperature to the logits in place beforehand (`apply_temperature=False`), so this task takes
temperature-scaled logits and uses `temperature` only to separate greedy from stochastic rows.

## Semantics a candidate must keep

- The random stream is part of the contract (the sampled token must be the one the reference draws).
  The reference implements Philox-4x32-10 with integer arithmetic; seeds and positions use their
  64-bit values (positions up to 2^40 exercise the counter's high word).
- The noise is `-log(-log1p(-u))` with `u` clamped to at least `4.6566127342e-10`. The naive
  `-log(-log(u))` evaluated in float32 quantizes the winning tail (`1 - u` near 2^-24) and picks
  different tokens, so it fails the exact comparison. Ties are broken toward the lowest index.
- `compare.out` is `exact` (the non-default bound the auditor warns about): the output is a token id, so any
  tolerance would accept a wrong token. Rows are independent. The only residual
  risk is a float32 near-tie between the two best noisy logits (probability about 1e-6 per row)
  that float32 and float64 round differently.

## Workloads

`n_tokens` is the number of sampled rows: decode batches, speculative positions and prompt-logprob
chunks. Rows of 64 to 6400 are timed (100x span; the largest case holds 3.9 GB of logits); smaller
batches run under the 10 us floor of the 600 KB-per-row read and are correctness-only. The
boundary correctness cases are 1, 17, 33 and the unaligned 1000.

## Baseline

vLLM's two-stage Triton kernel: one program per (row, 1024-wide block) writes the block winner and
its value, then the winners are combined per row. Upstream combines them with `torch.argmax` and
`gather`; PyTorch compute is not allowed inside `run()`, so that step is a small Triton kernel
(`_combine_blocks_kernel`). The processed-logits and fp64 variants were dropped. Provenance and
modifications are in `task.yaml`.
