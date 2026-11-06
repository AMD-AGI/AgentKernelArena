# Sqrt-softplus top-k routing, 384 experts, top-8, float32 logits

Routes each token to 8 of 384 experts. Scores are `sqrt(softplus(logit))` (softplus is the identity
above 20, as in PyTorch and AITER); the selection score is the score plus `correction_bias`; the 8
largest selection scores win. Weights are the unbiased scores of the chosen experts, renormalized to sum to 1 and multiplied by `route_scale` (2.5). The order inside a row is unspecified, so rows are
compared sorted (`element_order: any_in_full_rows`).

Shapes follow the AITER `topk_softplus` fused routing op (`aiter.topk_gating(score_func="sqrtsoftplus")`).
Router logits are float32; the correction bias is float32. Token counts range from decode batches to
batched prefill; decode-sized batches and small prefill chunks run below the 10 us timing floor and are
correctness-only. 384 is not a power of two, so the baseline pads the expert axis to 512 lanes and masks them. Sibling scenarios cover 256 experts, 128 experts with top-4, and bfloat16 logits.

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

The routing kernel is cheap per token (about 1536 bytes read per token), so only the large prefill token counts
are timed; the smallest performance case (2560 tokens) is kept to satisfy the 100x span and relies on the
kernel being latency-bound. The MI355X QA audit found every performance case at or above the 10 us floor.

## Comparison bounds

`topk_ids` is compared exactly (integer output, rows sorted first). Because the inputs have no
decision-boundary near-ties, any id difference is a genuine routing error.

`topk_weights` uses `sqnr` with `min_db: 130` instead of the baseline-relative default. Each weight is a score
divided by the sum of its row's 8 scores, so the float32 rounding of that sum, of the division and of the scaling
moves all 8 weights of a row together, and the per-element error depends on the implementation's exp, log and sqrt.
In the MI355X QA audit a float32 PyTorch evaluation of the reference (a correct float32 implementation with another
summation and division order) exceeded the default bound on 0.16% to 0.24% of the weights in 7 of 10 cases (t1000:
16 of 8000; t262144: 4677 of 2097152; peak 1.38x to 2.37x, bias z at most 0.015). An earlier
`max_violation_fraction: 0.01` admitted those, but 1% of the 8, 24 and 512 weights of t1, t3 and t64 admits zero,
zero and five elements, and a CPU emulation under the current comparator shows that it still rejects correct designs
there. The emulation evaluates the baseline's formula with GPU-style `exp(x) = exp2(fl(x log2 e))` and
`log(v) = log2(v) ln 2`, either correctly rounded or with exp2, log2 and sqrt each off by up to 1 ulp. These two
bracket the audit: from 1000 tokens on, the float32 PyTorch evaluation exceeds the bound on a median 0.61% and
0.09% of the weights against them, while the audit measured 0.16% to 0.24%. Share of draws that
`max_violation_fraction: 0.01` rejects (1-ulp baseline emulation / correctly rounded one; 400 draws per case for
t1, t3 and t64, 70 per case for 1000, 2560 and 8192 tokens):

| correct float32 design | t1 | t3 | t64 | 1000 tokens and more |
| --- | --- | --- | --- | --- |
| reference in float32 PyTorch | 0.5% / 3.5% | 1.8% / 13% | 0% / 12% | 0% / 0% |
| AITER kernel formula `log2(1 + exp2(x log2 e)) ln 2`, sequential sum, `route_scale / sum` | 1.5% / 2.5% | 0.8% / 9.3% | 0.3% / 12% | 0.5% / 1.4% |
| the same with exp2, log2 and sqrt off by up to 1 ulp | 13% / 35% | 27% / 70% | 72% / 100% | 100% / 100% |

From 1000 tokens on the last design exceeds the bound on a median 1.5% / 5.6% of the weights. Rows of
correlated errors also weaken the default bound's bias test (it counts every element as an independent sample);
in the MI355X audit of the 256-expert float32 sibling, the same float32 evaluation failed it (bias z 0.044 in t2560
against an allowance of 5/sqrt(20480) = 0.035).

`sqnr` compares the norm of the whole output's error (against the reference rounded to float32) with the norm of
the reference and has no bias test. CPU emulation with the task's input distribution (40 draws each of 64, 1000,
2560 and 8192 tokens and 16 draws of 16384 tokens for the first column, 262144 rows for the single-row column,
440 draws of t1):

| float32 implementation | SQNR, 64 tokens and more | worst single row | worst t1 draw |
| --- | --- | --- | --- |
| reference in float32 PyTorch (`logaddexp`, sum, divide, scale) | 142.4 to 144.1 dB | 133.9 dB | 136.5 dB |
| `log1p(exp(x))`, sequential sum, multiply by 2.5 / sum | 142.4 to 144.2 dB | 132.6 dB | 134.8 dB |
| baseline formula, GPU-style exp2 and log2, sequential sum | 142.0 to 143.4 dB | 133.6 dB | 136.2 dB |
| AITER kernel formula, sequential sum, `route_scale / sum` | 142.1 to 143.9 dB | 132.6 dB | 136.1 dB |
| the same with exp2, log2 and sqrt off by up to 1 ulp | 139.5 to 140.9 dB | 132.5 dB | 135.3 dB |
| baseline formula, exp, log and sqrt off by up to 1 ulp | 139.8 to 141.2 dB | 133.3 dB | 135.5 dB |
| baseline formula, exp, log and sqrt off by up to 2 ulp | 136.9 to 138.0 dB | 130.6 dB | 133.0 dB |
| baseline emulation, GPU-style exp2 and log2 | 142.7 dB or more | 134.7 dB | 139.1 dB |
| baseline emulation, exp2, log2 and sqrt off by up to 1 ulp | 139.5 dB or more | 133.4 dB | 135.5 dB |

The 130 dB threshold is at least 3.0 dB below every emulated t1 draw of every design and at least 6.9 dB below
them from 64 tokens on; no single row of any design falls under it. The baseline faces the same bound in its own
correctness check, with a margin of at least 3.4 dB per row and 9.5 dB from 64 tokens on. The threshold rejects the
`shifted_16_ulp` control (output scaled by 1 + 16 u) on the weights alone in every case: at most 120.5 dB from 64
tokens on, at most 121.5 dB in a t1 draw and 122.1 dB in the best single row. A uniform 6-ulp scaling is rejected
from 64 tokens on (128.4 to 128.9 dB); a 4-ulp scaling (131.9 to 132.5 dB) passes. fp8 rounding gives at most
45 dB in any single row, and zeroed, half-written, stale or unwritten (NaN) outputs fail outright. A localized error is caught when
it is large against the whole output: at 16384 tokens one row scaled by 1.01 gives 82 dB, one zeroed row 42 dB
and a single weight off by 3e-4 relative 121 dB, while a single weight off by 1e-4 (131 dB) passes; at 262144
tokens the signal is 12 dB larger (one row scaled by 1.01 about 94 dB, one zeroed row about 54 dB).
