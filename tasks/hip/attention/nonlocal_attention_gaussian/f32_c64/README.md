# nonlocal attention gaussian, float32, 64 channels

## Operator

Non-local attention block, embedded Gaussian variant: softmax over the key positions of `q^T k` (plain `exp`, no max subtraction, as in the original module), `y = v @ energy`, `out = Wg y + bg`. Input `x` is `[batch, 64, hw]` float32 (hw = H*W flattened); the three 1x1 convolutions are
`[out, 64]` matrices with bias; the 1x1 output convolution has 64 channels. q and k have 16 channels
(channels / 4, the reduction ratio of the original module), v has 64. Output is float32 `[batch, 64, hw]`.

## Production origin and scenario

Non-local blocks (Wang et al., "Non-local Neural Networks") inserted into image and video networks
at 32x32 to 128x128 feature maps. The previous version of this task (hip2hip/NormalAttention_embedded_gaussian) used 4 channels and
hw up to 4096, which is a toy shape; channels are rescaled to 64 (the width such blocks use) and hw to
1024-16384. Workloads: 8 performance cases span B*hw = 1024 to 131072 (128x); the cost is about
160 * batch * hw^2 flops, so even the smallest case is far above the 10 us floor. Eight small, odd and
minimum shapes (hw=1, 16, 50, 100, 333, 1000, 1025, 2049; batch 1, 3, 5, 17, 2) are correctness-only.
Weights are N(0, 0.0625) (0.125 for the output convolution) and biases N(0, 0.1), so the energies are of order one
and `exp` cannot overflow in float32 (energies stay within about +-6, so softmax denominators are at most ~16384*e^6).

## Baseline

A HIP baseline, task-original: the previous task provided a hipBLAS/ATen-convolution composition (not
allowed as a baseline here: no BLAS, no ATen compute, and it materializes the hw x hw energy). This
baseline is a rewrite of it in plain HIP: a fused q/k/v projection kernel, a row-sum kernel computing the softmax denominators, and a tiled attention kernel that recomputes exp and applies the output convolution in its epilogue, never storing the energy. Provenance is the in-repository hip2hip task at the recorded revision; the kernel
is float32 throughout with fp32 accumulation. Channel counts are compile-time constants (64/16), which
is why the scenario fixes them.

## Reference and bound

`reference.py` evaluates the block in float64, chunking the summed position to bound memory. It was
checked on CPU against the original functional module.

`out` uses `baseline_relative` with `k: 8` and `max_violation_fraction: 0.002` instead of the default
`k: 2` with no violations (a bound override).

The baseline is not more accurate than float32 accumulation. It is plain float32 with FMA chains in
one fixed order: each projection starts from its bias and adds 64 products, each energy is a 16-term
dot product, each softmax denominator is summed in four interleaved slices of the j axis, the weight
is `exp(e) * fl(1 / s)`, the summed position i is accumulated sequentially in one register per output
(hw terms) and the output convolution starts from its bias and adds 64 products. Its error is 3 to
9 ulp RMS (relative to the mean |out|) for hw up to 333 and grows with the sequential sum to 14 to
18 ulp at hw = 4096 and 23 to 28 ulp at hw = 16384 (SQNR from about 131 dB down to 117 dB). Any other
correct float32 implementation has independent rounding noise of the same size, and the default bound
rejects it. On MI355X the QA calibration probe (the reference evaluated in float32 PyTorch) failed all
16 cases with `k: 2`: 12.5% of the 64 elements of `b1_n1`, 2.1% of `b1_n4096`, 0.55% of `b8_n1024` and
less elsewhere, by up to 5.3 times the bound.

The new bound comes from a CPU study under the current comparison (rows of fewer than 8 elements use
the whole output's RMS; the bias fallback scales with 1/sqrt(n); admitted elements must stay within 16
bounds). It compared a float32 emulation of the baseline's exact arithmetic with the float64
reference, together with 18 independent correct float32 implementations:

- the reference evaluated in float32 PyTorch;
- MFMA-like k=4 chunked sums in every product, with the biases added last, alone and combined with a
  one-pass max-subtracted softmax using `exp2(e * log2(e))` and a division (a typical Triton design);
- no FMA contraction; the projection and output-convolution biases added after the dot products;
- for the sum over i: the sequential sum reversed, 4- and 32-way interleaved partial sums, 256-position
  blocks, k=4 chunks and pairwise summation;
- for the denominators: one sequential chain, 64-wide block sums, and a one-pass max-subtracted sum;
- `exp2(e * log2(e))` instead of `exp`; division by the denominator; the reciprocal folded into v;
- the output convolution folded into v (`out = (Wg v) energy + bg`).

It covered all 16 workload shapes with 2 to 316 draws each (13166 samples of a correct
implementation). Every correctness case and `b1_n1024`, `b1_n4096`, `b4_n2048` and `b8_n1024` were
evaluated in full (except two draws of each of the last two, on 512 positions); the four largest used
every image and 384 or 512 of the output positions, with the bias test applied at the full case's
size. Results:

- `k: 2` failed 7027 samples, all elementwise (the float32 reference alone 466 of 769), with up to
  11% of the elements out of bound.
- `k: 4` failed 2419 samples and `k: 6` 785. `k: 8` with no allowance failed 307 elementwise: one to
  three elements of the 1024 of `b1_n16`, and at most 2.9e-4 of the elements in the other cases.
- `k: 8` with 0.002 failed one sample elementwise: 3 of the 1024 elements of `b1_n16`, with the
  biases added last, in 1 of 316 draws. The float32 reference itself put 2 elements of `b1_n16` out
  of bound in 2 of its 316 draws, which is why 0.001 is not enough. Admitted elements stayed within
  6.6 bounds (2.5 except for the folded output convolution), far below the 16-bound limit. `b1_n1`
  passed all 5688 samples.

The bound still rejects every QA control. In the emulation a 16-ulp relative shift of the baseline
output was rejected in 707 of 769 draws, in every draw of `b3_n100`, `b17_n50` and `b2_n1000`; at
hw <= 100 its median violation fraction is 0.67% to 3.1% (at least 0.45% in every draw of `b3_n100`
and `b17_n50`), 2 to 15 times the allowance. Float8 rounding was rejected in every draw (more than
99.9% of the elements out of bound), and rounding v and the attention weights to TF32 in every draw
(SQNR at most 100 dB).

Residual risks that no allowed `k` or `max_violation_fraction` removes:

- The bias test. Each attention weight row sums to one over j, so the mean over j of `y[c, j]`
  equals the mean over i of `v[c, i]`, and the mean error of the whole output is set by the mean of
  the v projection's rounding errors: one random number per draw, independent between two
  implementations that round v differently, while the rest of the error averages out. When the
  baseline's mean error happens to be small, a correct implementation fails the fallback
  (`|mean| <= 8 |baseline mean| + 5 rms / sqrt(n)`). With this bound the bias test alone rejected
  4 of 74 performance-case samples of the float32 reference and of the k=4-chunked design, one
  sample each of three other designs that change the projection or output-convolution arithmetic
  (of 22 to 74), and 2 of 695 small-case samples of the single-chain denominator; no other failure
  remained. At about 5% per performance-case draw, a design that rounds v like a GEMM is rejected in
  a sizeable share of full evaluations (8 performance cases, 2 draws each). A `sqnr` bound
  avoids that test but cannot replace it here: correct designs reach 117.1 dB at `b2_n16384` (the
  baseline itself 117.5 dB), while the shifted control stays at or above 119.1 dB in every case with
  hw <= 2049 and at 115.7 to 117.8 dB in the two largest, so a threshold that accepts the correct
  designs with any margin lets the control pass.
- On MI355X the float32 reference was less accurate than its CPU emulation in `b1_n4096`, `b8_n1024`,
  `b4_n2048`, `b3_n2049` and `b8_n4096` (2.1% and 0.55% of the elements over the `k: 2` bound against
  at most 0.01% on CPU), so this study does not predict its probe result there. Scaling its `k: 2`
  peaks (5.1 and 5.3) by 2/8 puts them near 1.3 bounds, so a few elements of those cases may still
  exceed the bound.

## Risks for the GPU audit

The MI355X QA audit of the previous revision (default bound) passed baseline compile, correctness,
CUDA-graph timing of every performance case above the 10 us floor and the full control suite; its
calibration probe is the reason for the bound override above. The attention kernel computes `exp`
twice per energy element (row-sum and use); this is a design choice of the baseline, not of the task.
