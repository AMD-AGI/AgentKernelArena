# apply_penalties, float32 logits, vocabulary 151936, speculative decode

In-place repetition, frequency and presence penalties of the logits of a speculative-decode step.
Each request contributes a group of 4 consecutive rows (one target token and 3 draft tokens). A row
at group position `p` counts, besides the request's generated-token counts `output_bin_counts`, the
`p` draft tokens before it in its group. Tokens present in the prompt (packed bitset
`prompt_bin_mask`) or with a positive count are penalized by `repetition_penalty` (divide when the
logit is positive, multiply otherwise); then `frequency_penalty * count` and
`presence_penalty * (count > 0)` are subtracted. Requests with all three penalties disabled leave their
rows untouched.

**Production origin.** vLLM's v1 GPU model-runner sampler
(`vllm/v1/worker/gpu/sample/penalties.py`, `_penalties_kernel` and `apply_penalties`). The baseline
is that Triton kernel, a grid of (row, 8192-wide column block). The checkout's kernel reads the draft
tokens with a dynamic loop over the row's local position; the older kernel of the retired task took
the speculative length as a compile-time constant and a `num_speculative_tokens` argument, which the
current operator no longer has.

**Scenario.** Vocabulary 151936 (Qwen2/Qwen3 family) with 3 draft tokens per request; the request count
is the only varying axis (4 rows per request, `n_tokens = 4 * n_reqs`). Request states are visited in
shuffled order. One request in 8 has every penalty disabled, one in 8 only frequency/presence off, one
in 8 only repetition off; the rest use all three (repetition 1.05-1.5, frequency 0.1-0.5, presence
0.1-0.6). Prompt bitsets hold about 3% of the vocabulary and output counts about 1% (values 1-7).

**Workloads.** 1 to 1024 requests (4 to 4096 rows). Fewer than 10 requests run under the 10 us timing
floor and are correctness-only; 10 to 1024 requests are the eight performance cases (a 102x span).

**Comparison.** `baseline_relative` against a float64 reference built from the same penalty definition
(counts accumulated in int64, bitset unpacked explicitly), with `k: 4` and `max_violation_fraction: 0.0005`
(the defaults are 2 and 0). The bound is each element's distance to the baseline's own error (or the row's
RMS of it) plus one output rounding. The baseline contracts `logits -= freq * count` into a fused
multiply-add and multiplies by a reciprocal, so almost all of its elements are exact or within one ulp,
while an equally correct kernel that rounds `freq * count` and the subtractions separately (or divides
instead of multiplying by the reciprocal) differs by an ulp of the intermediate. Where the penalized logit
cancels toward zero, that ulp is many output roundings and exceeds the tight bound at a sparse set of
elements. A float32 PyTorch evaluation of the reference failed the default bound in all 12 cases (violating
fraction up to 1.9e-4, peak 3.2x the bound). With `k = 4` the violating fraction of such an evaluation falls to
about 3e-5 and the peak to under 2x (a CPU emulation of the baseline and the float32 evaluation on the
r10 and r100 cases); `max_violation_fraction = 0.0005` leaves a margin of several times over the measured
fraction and is far below the 4 to 40% of elements a wrong penalty (a missed draft-token count, a
wrong bit, a skipped row class) changes. The shifted-by-16-ulp control, which moves every element, is still
rejected under this bound, and the default bias check is kept.
