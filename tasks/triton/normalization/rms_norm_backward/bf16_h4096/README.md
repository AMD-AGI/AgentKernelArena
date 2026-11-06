# RMSNorm backward, bfloat16, hidden size 4096

Backward of `y = x * rsigma * gamma` (rsigma = `rsqrt(mean(x^2) + epsilon)` per token), the training-time companion of
the forward RMSNorm in Llama-3 8B and Mistral 7B (hidden size 4096). Inputs are the output gradient `dy`, the saved
activation `x`, `gamma` and the per-token `rsigma` the forward saved. Outputs are `dx` (bfloat16, same shape as `x`)
and `dgamma` (float32, the sum over all tokens of `dy * x * rsigma`).

`rsigma` is an input, not recomputed: the task-owned generator (`inputs.py`) derives it from `x` in float32 with
epsilon 1e-5, so it is consistent with `x` as the forward would have saved it. The reference (float64, chunked over
tokens) uses the given `rsigma`. The old task also covered a zero-centered gamma variant (`gamma + 1`); that variant is
not part of this scenario.

## Baseline

AITER's Triton RMSNorm backward at commit `652f2d4b768239aefd515c929e84817681ff598d`: `_rmsnorm_bwd_triton`, a
persistent kernel with one program per compute unit (`NUM_PRGMS = min(n_tokens, CU count)`, 8 warps, one 4096-wide
block per row) that writes `dx` and accumulates a per-program row of partial `dgamma` in registers, and
`_rmsnorm_bwd_dg_reduce_triton`, which sums the partial rows into `dgamma`. Both kernels are unchanged; `run()` is
the reduced `_rmsnorm_backward` launcher. The partial-dgamma scratch (`NUM_PRGMS x 4096` float32) is allocated inside
`run()`. The old task carried an earlier AMD copy of the kernel that wrote a full `n_tokens x 4096` float32 `dg_tmp`
(2 GB for the largest case); the production AITER form is used instead because it is the code that ships, and the old
public output `dg_tmp` is an intermediate, not an operator output.

## Comparison

`dx` uses the default bound. `dgamma` uses `baseline_relative` with `k: 4` instead of 2: it is a sum over up to 262144
tokens, so its float32 rounding error depends on the reduction order (the baseline reduces per program and then across
programs), and a candidate that orders the sum differently legitimately differs from the baseline by a few times the
baseline's own error. The baseline must still reach 25 dB SQNR against the float64 reference.

The QA calibration evaluated the reference in float32 (one fixed summation order over the tokens) and found that it
exceeds the `k: 4` bound for a small share of the 4096 `dgamma` elements: a violation fraction of 0.00464 in `t1000`
(and violations in `t1024`, `t2048` and `t32768`), at a peak of 1.36x the bound and a bias z-score of 0.012. The
baseline's two-level reduction (per program, then across the 256 programs) is more accurate than a single float32
accumulation order, so its error RMS, which scales the bound, is smaller than the error of a correct candidate that
accumulates in float32 in another order; the excess lands on a few elements, with no bias. `max_violation_fraction: 0.01`
admits that rate with about 2x headroom, while the peak limit (16x the bound) still rejects a skipped tail or zeroed
tile, and the bias test and the 25 dB SQNR requirement on the baseline are unchanged.

## Workloads

Per token the kernel moves about 24 KB (`x`, `dy` in and `dx` out, bfloat16), so cases need about 1.2k tokens to exceed
the 10 us timing floor. Performance cases start at 2048 tokens and span 128x up to 262144 tokens (6.3 GB moved); smaller
and unaligned batches (1, 3, 48, 1000, 1024) are correctness-only. The float64 reference is computed in chunks of 32768
tokens.
