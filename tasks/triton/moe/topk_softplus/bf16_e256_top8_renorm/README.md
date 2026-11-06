# Sqrt-softplus top-k routing, 256 experts, top-8, bfloat16 logits

Routes each token to 8 of 256 experts. Scores are `sqrt(softplus(logit))` (softplus is the identity
above 20, as in PyTorch and AITER); the selection score is the score plus `correction_bias`; the 8
largest selection scores win. Weights are the unbiased scores of the chosen experts, renormalized to sum to 1 and multiplied by `route_scale` (2.5). The order inside a row is unspecified, so rows are
compared sorted (`element_order: any_in_full_rows`).

Shapes follow the AITER `topk_softplus` fused routing op (`aiter.topk_gating(score_func="sqrtsoftplus")`).
Router logits are bfloat16; the correction bias is float32. Token counts range from decode batches to
batched prefill; decode-sized batches and small prefill chunks run below the 10 us timing floor and are
correctness-only. Router logits are bfloat16 (the dtype of the original task) with a float32 correction bias; the baseline widens them to float32 in the kernel.

## Inputs

`inputs.py` draws logits from `N(0, 1.5^2)` and redraws the rows whose margin between the 8th and
(k+1)th selection score is below 1e-4. The production op tolerates such near-ties; an exact id comparison
cannot, because float32 rounding or a fast exp/log in a kernel legitimately differs from the float64
reference there. The correction bias is `N(0, 0.1^2)`.

## Baseline

Triton rewrite of the semantics of ROCm/aiter's `topk_softplus` (a HIP kernel with no Triton form
upstream): one program per token, iterative arg-max selection over a register vector, softplus evaluated
accurately for very negative logits. No upstream code is copied. The repository's MIT license is in
`baseline/LICENSE`.

The routing kernel is cheap per token (about 512 bytes read per token), so only the large prefill token counts
are timed; the smallest performance case (2560 tokens) is kept to satisfy the 100x span and relies on the
kernel being latency-bound. The MI355X QA audit found every performance case at or above the 10 us floor.

## Comparison bounds

`topk_ids` is compared exactly (integer output, rows sorted first). Because the inputs have no
decision-boundary near-ties, any id difference is a genuine routing error.

`topk_weights` uses `sqnr` with `min_db: 130` instead of the baseline-relative default. Each weight is a score
divided by the sum of its row's 8 scores, so the float32 rounding of that sum, of the division and of the scaling
moves all 8 weights of a row together. In the MI355X QA audit a float32 PyTorch evaluation of the reference (a
correct float32 implementation with another summation and division order) exceeded the default bound on 0.15% to
0.39% of the weights in 8 of 10 cases (t64: 2 of 512; t262144: 3886 of 2097152; peak 1.11x to 1.93x). A violation
allowance would not make the bound safe, because the default bound's bias test fails correct designs here. The
bfloat16 logits have 8 significant bits, so the scores of a row's top-8 experts come from a small set of float32
values, and the roundings of the row sum do not average out over the rows: they leave a mean error that depends on
the summation order. A CPU emulation with the task's input distribution gives a design that sums the 8 weights
sequentially (the float32 PyTorch evaluation on the CPU, or the AITER kernel formula) a mean weight error of about
0.1 of its RMS error. The bias statistic of the default bound (mean error times sqrt(n) over RMS error, allowed up
to 5) then grows with the token count: 3.2 at 64 tokens, 9.2 at 1000, 15 at 2560 and 28 at 8192 (the float32
evaluation on the GPU, with another summation order, stayed at 3.2 or below in the audit). The emulation evaluates
the baseline's formula with GPU-style `exp(x) = exp2(fl(x log2 e))` and `log(v) = log2(v) ln 2`, either correctly
rounded or with exp2, log2 and sqrt each off by up to 1 ulp (from 1000 tokens on, the float32 PyTorch evaluation
exceeds the bound on a median 0.79% and 0.10% of the weights against them, bracketing the audit's 0.15% to 0.22%).
With `max_violation_fraction: 0.01`, sequential-sum designs still fail against the 1-ulp baseline emulation in 75%
to 85% of 40 draws at 1000 tokens and in every draw at 2560 and 8192 tokens (43% to 60%, 88% to 95% and 100%
against the correctly rounded one), almost all on the bias test; with `k: 4` none of 20 draws of 8192 or 32768
tokens passes for the float32 PyTorch evaluation or the AITER formula against the 1-ulp emulation. The bound also
rejects 24% to 34% of the t64 draws of such designs against the correctly rounded emulation (400 draws), where 1%
of 512 weights admits only five elements.

`sqnr` compares the norm of the whole output's error (against the reference rounded to float32) with the norm of
the reference and has no bias test. CPU emulation with the task's input distribution (40 draws each of 64, 1000,
2560 and 8192 tokens and 16 draws of 16384 tokens for the first column, 262144 rows for the single-row column,
440 draws of t1):

| float32 implementation | SQNR, 64 tokens and more | worst single row | worst t1 draw |
| --- | --- | --- | --- |
| reference in float32 PyTorch (`logaddexp`, sum, divide, scale) | 142.5 to 143.9 dB | 133.5 dB | 137.2 dB |
| `log1p(exp(x))`, sequential sum, multiply by 2.5 / sum | 142.6 to 144.3 dB | 133.3 dB | 135.3 dB |
| baseline formula, GPU-style exp2 and log2, sequential sum | 142.3 to 143.8 dB | 133.8 dB | 136.7 dB |
| AITER kernel formula `log2(1 + exp2(x log2 e)) ln 2`, sequential sum, `route_scale / sum` | 142.4 to 144.1 dB | 133.3 dB | 134.9 dB |
| the same with exp2, log2 and sqrt off by up to 1 ulp | 139.4 to 140.6 dB | 132.7 dB | 133.6 dB |
| baseline formula, exp, log and sqrt off by up to 1 ulp | 140.1 to 141.2 dB | 133.1 dB | 135.5 dB |
| baseline formula, exp, log and sqrt off by up to 2 ulp | 137.0 to 138.1 dB | 130.4 dB | 132.7 dB |
| baseline emulation, GPU-style exp2 and log2 | 143.2 dB or more | 136.2 dB | 136.9 dB |
| baseline emulation, exp2, log2 and sqrt off by up to 1 ulp | 139.9 dB or more | 133.4 dB | 135.5 dB |

The 130 dB threshold is at least 2.7 dB below every emulated t1 draw of every design and at least 7.0 dB below
them from 64 tokens on; no single row of any design falls under it. The baseline faces the same bound in its own
correctness check, with a margin of at least 3.4 dB per row and 9.9 dB from 64 tokens on. The threshold rejects the
`shifted_16_ulp` control (output scaled by 1 + 16 u) on the weights alone in every case: at most 120.5 dB from 64
tokens on, at most 121.5 dB in a t1 draw and 121.8 dB in the best single row. A uniform 6-ulp scaling is rejected
from 64 tokens on (128.4 to 128.9 dB); a 4-ulp scaling (131.9 to 132.5 dB) passes. fp8 rounding gives at most
45 dB in any single row, and zeroed, half-written, stale or unwritten (NaN) outputs fail outright. A localized
error is caught when it is large against the whole output: at 16384 tokens one row scaled by 1.01 gives 82 dB, one
zeroed row 42 dB and a single weight off by 3e-4 relative 121 dB, while a single weight off by 1e-4 (130.2 dB)
passes; at 262144 tokens the signal is 12 dB larger (one row scaled by 1.01 about 94 dB, one zeroed row about
54 dB).
