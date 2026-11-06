# topk_topp_mask, float32 logits, vocabulary 151936

In-place per-row top-k followed by top-p filtering of the logits of each sampled row. A row keeps its
`k` largest logits (all of them when `k >= vocab_size`); among those it keeps the smallest set of
largest logits whose softmax mass, renormalised over the `k` survivors, reaches `p` (all of them when
`p >= 1`). Every dropped logit becomes `-inf`; kept logits are unchanged. `k` and `p` are per row.

**Production origin.** vLLM's v1 sampler (`vllm/v1/sample/ops/topk_topp_triton.py`), the Triton
implementation of the algorithm of "Qrita" (Park et al., arXiv:2602.01518): a Gaussian-quantile pivot
estimate gathers a truncated set of candidate logits, then a pivot search over logits (top-k) and
probabilities (top-p) finds the cut, and a last pass masks the row. One program handles several rows
in turn and keeps a scratch row of probabilities.

**Scenario.** Vocabulary 151936 (Qwen2/Qwen3 family), float32 logits; the number of rows is the only
varying axis. Rows cycle through four kinds: top-k only (`p = 1`), top-p only (`k = vocab_size`), and
both filters (two of four), with k in {1, 8, 20, 50, 100, 400, 2000, 10000} and p between about 0.5 and
0.99; one row in 16 has both filters disabled and is unchanged.

**Baseline.** The vLLM kernels, vendored from revision `edbc4969a76b25c531f7b9bb16b79984e65ef023`
with the `run()` entry replacing the host wrapper (which cached the kernel's tables on the device and
converted `k` and `p`; none of that is allowed in a candidate call). `run()` launches
`min(num_SMs, n_tokens)` programs, allocates one scratch row of probabilities per program, and enables
both filters, as vLLM does when any request of the batch uses either. The kernel is not a rewrite.

**Lookup tables as inputs.** The kernel reads two 200-entry constant tables (Gaussian sigma by
percentile) that the vLLM wrapper builds on the device once per process. A launch inside a timed call
cannot create device constants, so they are declared as inputs (`role: weight`, produced by
`inputs.py`, copied from the same vLLM file). The reference does not use them and an implementation may
ignore them.

**Workloads.** Decode batches of 1 to 1024 rows and a 4096-row prompt-logprob chunk. One program
streams a whole 152k-element row several times, so every batch from 16 rows is expected above the
10 us floor on the baseline. Batches 16 to 4096 are the eight performance cases (a 256x span); batches
1, 3, 17, 24 and 100 are correctness-only and cover the axis minimum, odd and unaligned sizes and a
row with both filters disabled. Memory of the largest case is
2.5 GB of logits per input draw.

**Comparison.** `logits` compares `exact`: every element of the result is either the input value or
`-inf`, so a difference is a wrong cut, not rounding. The vLLM kernel is a pivot-search approximation:
on arbitrary logits it may differ from the exact selection by elements adjacent to the cut. The input
generator therefore makes every cut unambiguous so that exact selection is well defined for any
float32 implementation: after the k-th largest logit, the following logits are lowered until the gap is
at least 0.05; for top-p, the cut element is chosen with probability at least 1e-4 among the
candidates, the logits after it are lowered until the gap is at least 1.0, and `p` is set in the middle
of the cut element's probability interval, so the float32 partial sums of any accumulation order
(error about 1e-7) stay at least 5e-5 from the threshold. Logits are normal with standard deviation 4
before these adjustments (a peaked, LM-like distribution). Ties at a cut cannot occur.

**GPU risks to check in the QA audit.** The baseline is expected to pass the exact comparison given the
margins above, but this was designed from the kernel's source on CPU only; the pivot bisection stops
after 18 iterations, which resolves probability windows of about 4e-6 (the generator guarantees at least
6e-5 for the cut element at the widened gap). If the audit shows the baseline failing a few rows, the
margins in `inputs.py` (`_GAP_P`, `_P_MIN_PROB`) are the knobs.
