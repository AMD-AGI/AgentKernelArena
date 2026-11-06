# Sqrt-softplus top-k routing, 256 experts, top-8, float32 logits

Routes each token to 8 of 256 experts. Scores are `sqrt(softplus(logit))` (softplus is the identity
above 20, as in PyTorch and AITER); the selection score is the score plus `correction_bias`; the 8
largest selection scores win. Weights are the unbiased scores of the chosen experts, renormalized to sum to 1 and multiplied by `route_scale` (2.5). The order inside a row is unspecified, so rows are
compared sorted (`element_order: any_in_full_rows`).

Shapes follow the AITER `topk_softplus` fused routing op (`aiter.topk_gating(score_func="sqrtsoftplus")`).
Router logits are float32; the correction bias is float32. Token counts range from decode batches to
batched prefill; decode-sized batches and small prefill chunks run below the 10 us timing floor and are
correctness-only. The sibling scenarios `f32_e384_top8_renorm`, `f32_e128_top4_plain` and `bf16_e256_top8_renorm` cover the 384-expert, top-4 non-renormalizing and bfloat16-logit variants of the original task.

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

The routing kernel is cheap per token (about 1024 bytes read per token), so only the large prefill token counts
are timed; the smallest performance case (2560 tokens) is kept to satisfy the 100x span and relies on the
kernel being latency-bound. The MI355X QA audit found every performance case at or above the 10 us floor.

## Comparison bounds

`topk_ids` is compared exactly (integer output, rows sorted first). Because the inputs have no
decision-boundary near-ties, any id difference is a genuine routing error.

`topk_weights` uses `sqnr` with `min_db: 130` instead of the baseline-relative default. Each weight is a score
divided by the sum of its row's 8 scores, so the float32 rounding of that sum (and of the row's division and
scaling) moves all 8 weights of a row together. The element errors are therefore not independent, but the
default bound's bias test counts every element as an independent sample (mean error within 5/sqrt(n) of the error
RMS). In the MI355X QA audit a float32 PyTorch evaluation of the reference, a correct float32 implementation with
another summation and division order, stayed within the per-element bound on all but 0.17% to 0.39% of the weights
(peak 1.47x to 2.08x, admitted by the earlier `max_violation_fraction: 0.01`) but failed the bias test in t2560: its
mean error was 6.3 such standard errors from zero (bias z 0.044 against an allowance of 5/sqrt(20480) = 0.035).
Over the eight cases of 64 tokens and more this normalized mean error had an RMS of 2.8 instead of about 1. A CPU
emulation (three float32 implementations against a float32 emulation of the baseline, 1024 to 8192 tokens, 560
draws each) gives an RMS of 1.6 to 2.1 and 2 bias-test failures of correct candidates in 1680 draws. No `k` or
violation fraction changes that test. The per-element bound also depends on the baseline's exp and log: against a
CPU emulation of the baseline with the CPU's exp and log, the float32 evaluation exceeds it on up to 0.85% of the
weights (limit 1%), and a candidate whose exp, log and sqrt are up to 1 ulp less accurate on 4.2% to 4.4%.

`sqnr` compares the norm of the whole output's error (against the reference rounded to float32) with the norm of
the reference and has no bias test. CPU emulation with the task's input distribution (524288 rows for the global
and single-row figures, 200 draws of t1, all ten cases):

| float32 implementation | SQNR, 64 tokens and more | worst single row | worst t1 draw |
| --- | --- | --- | --- |
| reference in float32 PyTorch (`logaddexp`, sum, divide, scale) | 142.6 to 143.1 dB | 134.1 dB | 136.7 dB |
| baseline formula, CPU exp and log, tree sum | 143.4 to 143.9 dB | 135.3 dB | 138.7 dB |
| `log1p(exp(x))`, sequential sum, multiply by 2.5 / sum | 143.0 to 143.3 dB | 132.7 dB | 134.8 dB |
| GPU-style exp2(x log2 e) and log2(v) ln 2, each 1-ulp | 141.9 to 142.4 dB | 135.4 dB | 137.7 dB |
| exp, log and sqrt off by up to 1 ulp | 140.2 to 140.7 dB | 133.5 dB | 136.7 dB |
| exp, log and sqrt off by up to 2 ulp | 137.2 to 137.5 dB | 129.9 dB | 133.0 dB |

The 130 dB threshold is at least 3 dB below every emulated draw of every case for all these implementations and 7
to 14 dB below them from 64 tokens on; only the 2-ulp variant has single rows under it (2 in 10^6 rows, which
matters only for the 1- and 3-token cases). The emulated GPU-style baseline, with or without a 1-ulp sqrt, brackets
the audit's float32-probe violation fraction at 8192 tokens (0.08% to 0.33% against 0.19% measured) and has 140 to
142 dB, so the baseline's own check (it faces the same bound) has a margin of about 10 dB. The threshold rejects the
`shifted_16_ulp` control (output scaled by 1 + 16 u) in every case, including t1, by at least 8.9 dB (at most 121.1
dB) and a uniform 6-ulp scaling from 64 tokens on (128.6 to 128.8 dB); a 4-ulp scaling (132.0 dB) passes. fp8 rounding is at most 38 dB, and zeroed, half-written, stale or
unwritten (NaN) outputs fail outright. A localized error is caught when it is large against the whole output: in
the largest case (262144 tokens) one row scaled by 1.01 gives 94 dB and one zeroed row 54 dB, while a single weight
off by 3e-4 relative (133 dB) is not detected (at 1024 tokens a single weight off by 1e-4 gives 119 dB).
