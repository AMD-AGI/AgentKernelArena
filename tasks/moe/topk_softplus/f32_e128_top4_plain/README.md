# Sqrt-softplus top-k routing, 128 experts, top-4, no renormalization, float32 logits

Routes each token to 4 of 128 experts. Scores are `sqrt(softplus(logit))` (softplus is the identity
above 20, as in PyTorch and AITER); the selection score is the score plus `correction_bias`; the 4
largest selection scores win. Weights are the unbiased scores of the chosen experts multiplied by `route_scale` (1.0), without renormalization. The order inside a row is unspecified, so rows are
compared sorted (`element_order: any_in_full_rows`).

Shapes follow the AITER `topk_softplus` fused routing op (`aiter.topk_gating(score_func="sqrtsoftplus")`).
Router logits are float32; the correction bias is float32. Token counts range from decode batches to
batched prefill; decode-sized batches and small prefill chunks run below the 10 us timing floor and are
correctness-only. Weights are not renormalized and the route scale is 1.0 (the dense-MoE configuration of the original task).

## Inputs

`inputs.py` draws logits from `N(0, 1.5^2)` and redraws the rows whose margin between the 4th and
5th largest selection score is below 1e-4. The production op tolerates such near-ties; an exact id comparison
cannot, because float32 rounding or a fast exp/log in a kernel legitimately differs from the float64
reference there. The correction bias is `N(0, 0.1^2)`.

## Baseline

Triton rewrite of the semantics of ROCm/aiter's `topk_softplus` (a HIP kernel with no Triton form
upstream): one program per token, iterative arg-max selection over a register vector, softplus evaluated
accurately for very negative logits. No upstream code is copied. The repository's MIT license is in
`baseline/LICENSE`.

## Workloads

The routing kernel is cheap per token (about 512 bytes read per token), so only prefill-sized token counts
are timed. The MI355X QA audit measured the baseline (CUDA-graph, cold-cache median) at 9.9 us for 2560
tokens, 12.4 us for 8192, 24.6 us for 32768, 69.2 us for 131072 and 128.7 us for 262144: about 8.8 us
of fixed cost plus 0.45 ns per token. The 2560-token chunk is below the 10 us floor and is
correctness-only. The smallest performance case is 5120 tokens (about 11 us by that fit), and the
largest is 524288 tokens (about 245 us), which keeps the span at 102x. Counts above 131072 exceed a single
chunked-prefill batch; they correspond to batched or unchunked long-context prefill and are the only
way to keep a 100x span while every case stays above the timing floor. The largest case's logits occupy
268 MB.

## Comparison bounds

`topk_ids` is compared exactly (integer output, rows sorted first). Because the inputs have no
decision-boundary near-ties, any id difference is a genuine routing error. `topk_weights` uses the default
baseline-relative bound.
