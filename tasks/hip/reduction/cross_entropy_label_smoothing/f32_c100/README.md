# Label-smoothed cross entropy, float32, 100 classes, integer labels

For logits `x[rows, 100]`, int64 labels `t`, class weights `w` and smoothing `eps`:

```
lsm         = log_softmax(x, dim=-1) * w
losses[row] = -((1 - eps - eps/C) * lsm[t] + (eps/C) * sum_c lsm[c])     (0 where t == ignore_index)
out         = sum(losses) / (number of rows with t != ignore_index)
```

The weights multiply the log-probabilities (not a weighted mean), the mean divides by the count of
non-ignored rows, and `out` is 0 when every row is ignored. `ignore_index` is only active when it is
non-negative; the task uses `ignore_index = 0` with labels uniform on `[0, 100)`, so about 1% of
rows are ignored. `eps = 0.1`, `C = 100` and the class weights are uniform on `[0.5, 1.5]`. The
outputs are the float32 per-row losses (`reduction='none'`) and their float32 mean. The per-row
losses lie between about 0.9 and 20 (mean 6.3). Relative to PyTorch,
`losses = F.cross_entropy(x, t, weight=w, ignore_index=0, label_smoothing=eps, reduction='none')
- (eps/C) * F.nll_loss(log_softmax(x), t, weight=w, ignore_index=0, reduction='none')`: the original
kernel weights the target term by `1 - eps - eps/C`, not PyTorch's `1 - eps`.

Origin: the hip2hip `CrossEntropyLossLabelSmoothing` task (a gpumode label-smoothing cross entropy
module with the class axis last). Its manifest ran five `[N, C, H, W]` cases (last axis 8 to 32 as
the class axis) with float soft targets, a normalized class-ramp `smooth_dist`, `smooth_eps` from 0
to 0.6, no class weights and no ignore index. This task scores the integer-label path of the same
kernel instead (class weights, `ignore_index`, `smooth_eps`), the form used by classification and
segmentation training. The soft-target/`smooth_dist` branch of the original is not covered here: at
the original class counts (at most 32) a float target pair only reaches the 10 us floor beyond about
150,000 rows, so it would need its own scenario with very large row counts. The class count 100 is
the original task's largest channel count; the row counts are flattened batch x pixel positions of
per-pixel scoring (8x32x32 in the original task up to 256x256). Rows below 8192 would sit under the
10 us floor and are correctness-only, together with the odd and unaligned counts 3, 100 and 4097.

Why the per-row losses are an output: a single scalar cannot show that every row was computed. The
MI355X audit of the scalar-only version accepted a kernel that zeroes half of its work and one that
scales every result by `1 + 16 * 2^-24`, because a 16-ulp change of the mean is far smaller than the
legitimate summation-order differences of a float32 mean over up to 819,200 rows (below). The
original kernel already computed the per-row loss (and a valid flag) and reduced it afterwards,
`reduction='none'` is the per-position loss map of segmentation training, and the Triton
cross-entropy kernels of flash-attention and Liger-Kernel write one loss per row before averaging.
Writing it to the caller's `losses` instead of a scratch buffer moves no extra bytes; every row is
now checked on its own and `out` checks the reduction. (The same change was made to
`triton/reduction/cross_entropy_loss/f32_c4096`.)

Baseline: the provided HIP kernel, one thread per row (three passes over the 100 classes: max,
sum of exponentials, weighted log-probabilities) writing the per-row loss into `losses` and a valid
flag into a scratch buffer. The original launcher reduced them on the host side (`at::sum`, a host
read of the valid count, torch division), which a candidate may not do and which cannot be captured;
here a fixed 256-block partial reduction and a one-block finalize kernel produce the scalar in the
caller's `out` (deterministic order). Only the `smooth_dist=None` integer-label path of the original
kernel is kept.

## Bounds

Both outputs override the default `baseline_relative`. The numbers below come from CPU emulations
that round every float32 operation as the GPU does (IEEE round to nearest, FMA up to a negligible
double rounding), on this task's input distributions, scored as the evaluator scores them.

`losses`: `sqnr` with `min_db = 122` (whole-output signal-to-noise ratio against the reference
rounded to float32). A correct float32 row loss has errors of a few roundings of the log-sum-exp
and of the accumulated weighted log-probabilities. One natural formulation, the dot product of the
smoothed target `q_c = (eps/C) w_c + (1 - eps - eps/C) w_t [c == t]` with `lse - x_c` accumulated
by one sequential float32 accumulator per row, rounds 100 times at the loss's magnitude: its errors
reach 1.1e-5, and 20 `u = 2^-24` of the loss on single rows, more than the `shifted_16_ulp` control
(every element scaled by `1 + 16u`). No element-wise bound with zero violations separates the two,
and a violation fraction does not help: the tolerance bound then also requires an unbiased mean
error (`|mean| / rms <= 5 / sqrt(n)`), and the float32 rounding of the coefficient
`1 - eps - eps/C` biases every float32 implementation's losses by about -4e-8 (`|mean| / rms` of
0.03 to 0.09, against 0.0055 at 819,200 rows). An absolute tolerance of 3e-6 accepted the baseline's
order (peak 0.72 of the bound) but rejected that dot product (peak 3.1); the default bound rejected
even a tree-ordered, base-2 or `torch.float32` evaluation (peaks 1.35 to 1.57 at `k = 2`; the dot
product 7.7), and on the one-row case its error estimate is a single sample. The shift is
systematic while rounding errors average out, so whole-output SQNR separates them (6 draws of
131,072 rows, scored in blocks of the case sizes):

| Implementation | SQNR over 100 rows or more | single rows |
| --- | --- | --- |
| baseline order (with or without FMA contraction, `expf` or `exp2`-based, an exp with a 1-ulp error) | 141.0 dB lowest, median 143.4-144.3 dB | below 122 dB in at most 1.5e-5 of rows |
| tree or 64-lane reductions, online log-sum-exp with `sum w x - lse sum w`, base-2 log-sum-exp, the reference in `torch.float32` | 141.1 dB lowest | at most 7.6e-6 below 122 dB |
| smoothed-target dot product, one sequential accumulator (forward, reversed, FMA, noisy exp) | 130.2 dB lowest (133.4 dB from 1,000 rows) | median 137 dB; 5.0-5.5e-4 below 122 dB (3-row blocks: at most 7.4e-5) |
| smoothed-target dot product, tree | 137.8 dB lowest | at most 1.9e-6 below 122 dB |
| `shifted_16_ulp` control | 120.76 dB highest at 100 rows, 120.54 dB from 1,000 rows (120.41 dB nominal) | 97% below 122 dB |
| `smooth_eps / C` rounded to bfloat16 | 86 dB | |
| PyTorch's target coefficient `1 - eps`, the last class dropped from the sum, an unweighted smoothing sum | 60, 58, 53 dB | |
| ignored rows not zeroed | 20-26 dB | |

122 dB therefore rejects the shift in every case of 100 rows or more by at least 1.2 dB (1.4 dB from
1,000 rows) and accepts every correct float32 order there by at least 8 dB. The residual risk is in
the 1- and 3-row cases, where SQNR is the relative error of one or three rows: a correct sequential
dot-product kernel fails one of those four draws with a probability of about 1e-3, a kernel in the
baseline's order about 3e-5. Unwritten (NaN-poisoned) or non-finite elements always fail; the zeroed, half-written, stale
and fp8-rounded controls fail by tens of dB. A single wrong row fails above an error of about 1.7e-4
at 1,000 rows and 4.9e-3 at 819,200 rows; `out` independently rejects any change of the mean above
2e-4.

`out`: `tolerance` with `atol = 1e-6`, `rtol = 2e-4`. A bound proportional to the baseline's own
error on one number is not meaningful (that error can be arbitrarily close to 0). The baseline's
fixed-order two-stage float32 tree is within 1.4e-7 relative of the exact mean (50 emulated draws of
819,200 rows; at most 3.7e-4 of this bound in the MI355X audit). A candidate that accumulates the
row losses sequentially or with one float32 atomic per row in arbitrary order had errors up to
3.7e-5 relative (50 draws, 5 orders each), a factor 5.4 below the bound, and a warp tree followed by
atomics up to 4.1e-6, while a wrong divisor (all rows, or PyTorch's weight sum), a dropped tail or a
wrong smoothing term moves the mean by far more.

The reference runs in float64 in chunks of 131,072 rows. The largest case (819,200 rows) was not
run in the CPU smoke because it exceeds the pre-check's size limit; the reference was checked
against the original functional module on 5,000 rows and against PyTorch's `F.cross_entropy` with
the coefficient adjustment above (maximum difference 4e-15).

Measured on MI355X (QA audit of the scalar-only version, same kernels and bytes): the baseline takes
64 us at 8,192 rows, 65 us at 32,768, 183 us at 131,072, 1.88 ms at 524,288 and 3.12 ms at 819,200.

Risks for the GPU audit: the SQNR figures come from CPU emulation of the kernels' float32
arithmetic, not from the GPU; the 1- and 3-row cases carry the small residual risk above.
