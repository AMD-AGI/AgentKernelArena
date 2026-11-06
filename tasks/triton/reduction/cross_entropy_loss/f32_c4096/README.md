# Cross entropy, float32, 4096 classes

`losses[b] = logsumexp(logits[b]) - logits[b, targets[b]]` and `out = mean_b(losses[b])` for
`logits[batch, 4096]` float32 and int64 `targets` uniform on `[0, 4096)`: the values of
`F.cross_entropy(logits, targets, reduction="none")` and of `F.cross_entropy(logits, targets)` (mean),
in float32. Logits are normal with standard deviation 2, so the losses lie mostly between 2 and 20
(mean 10.3).

Origin: KernelBench level 1 problem 95 (CrossEntropyLoss) as used by the torch2hip task. Its cases
varied both batch and class count (8192x1024 up to 8192x16384, with `torch.rand` logits); this task
fixes the 32768 x 4096 scenario's class count and varies the batch. Batches below 2048 rows move
less than about 33 MB and sit near the 10 us floor, so they are correctness-only together with the
odd and unaligned counts 3, 100 and 1000. The largest performance case (204,800 rows) is 3.4 GB of
logits.

Why the per-row losses are an output: the original problem returns only the scalar mean. One
number cannot show that every row was computed: a kernel that writes half of its output is
indistinguishable from a correct one, and a 16-ulp error on the mean is smaller than the
legitimate summation-order differences of a float32 mean over 204,800 rows (below). Training loss
kernels materialize the per-token loss anyway (the Triton cross-entropy kernels of
flash-attention and Liger-Kernel and Megatron-LM's vocab-parallel cross entropy write one loss per
row and average afterwards), and the baseline already computed it into a scratch buffer. Writing it
to the caller's `losses` costs 4 bytes per 16 KB row; every row is now checked on its own and the
mean checks the reduction.

Baseline: a standalone Triton rewrite of what torch.compile (Inductor) generates for this operator:
one program per row computes the row loss with a one-pass (online) log-sum-exp over 1024-wide
tiles and a gather of the target logit and writes it to `losses`; a 256-program fixed-order partial
sum and a one-program finalize kernel produce the mean in `out`. It is not copied from a specific
Inductor output; provenance is recorded as the PyTorch source it replaces (BSD-3-Clause). The row
kernel masks the tail tile, so it is not tied to the 4096 class count.

Bounds (both `tolerance`, not the default `baseline_relative`):

- `losses`: `atol = 3e-6`, `rtol = 0` (the evaluator adds one float32 rounding, `2^-24 |y|`), about
  three float32 ulps at the loss magnitude (`ulp` on [8, 16) is 9.5e-7). The error of a correct
  float32 row loss is a few roundings of values near the log-sum-exp (about 10.3), not
  proportional to the loss, so the bound is absolute. A CPU emulation (4 draws of 16,384 rows)
  of the baseline's algorithm, `torch.logsumexp` in float32, two-pass max/sum, log-softmax then
  gather, an all-base-2 (`exp2`/`log2`) formulation and an exp with an extra 1-ulp error per
  call gave a worst error of 1.85e-6 (the base-2 form; 1.46e-6 for the others), at most 0.46 of
  the bound. Scaling the baseline's losses by `1 + 16 * 2^-24` violates the bound on 99.97% of the
  rows, by up to 4.6 times. The default bound is not used because its bias test is stricter than
  float32 permits here: at 204,800 rows it rejects a mean error above 0.011 of the error RMS
  unless the baseline's own mean error is at least half as large, while the float32 rounding of
  `log2(e)` in an `exp2(x * log2e)` exponential biases the losses by about +5e-8 (0.13 of the RMS)
  and a base-2 formulation by about -3e-8. Which of these the baseline's compiled `tl.exp`
  matches is a compiler detail, so a correct candidate with a different exponential could fail.
- `out`: `atol = 1e-6`, `rtol = 1e-4`. A bound proportional to the baseline's own error on one
  number is not meaningful. The baseline's fixed-order float32 tree reduction is within about
  1e-7 relative of the exact mean (at most 7e-4 of this bound in the MI355X audit); a candidate
  that accumulates the row losses sequentially or with one float32 atomic per row has errors up
  to 1.8e-5 relative at 204,800 rows (largest of 50 simulated draws, median 4e-6), which the
  bound accepts with a factor of 5 to spare, while a wrong divisor, a dropped tail or a missing
  target gather moves the mean by far more.

The reference runs in float64 in chunks of 16,384 rows to keep temporaries small.

Risks for the GPU audit: the error figures above come from a CPU emulation of the kernels'
float32 arithmetic, not from the GPU; the b2048 case baseline (19.7 us on MI355X) is the closest to
the 10 us floor.
