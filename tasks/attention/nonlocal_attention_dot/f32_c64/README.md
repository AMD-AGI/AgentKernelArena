# nonlocal attention dot, float32, 64 channels

## Operator

Non-local attention block, dot-product variant: `energy = elu(q^T k) / hw`, `y = v @ energy`, `out = Wg y + bg`. Input `x` is `[batch, 64, hw]` float32 (hw = H*W flattened); the three 1x1 convolutions are
`[out, 64]` matrices with bias; the 1x1 output convolution has 64 channels. q and k have 16 channels
(channels / 4, the reduction ratio of the original module), v has 64. Output is float32 `[batch, 64, hw]`.

## Production origin and scenario

Non-local blocks (Wang et al., "Non-local Neural Networks") inserted into image and video networks
at 32x32 to 128x128 feature maps. The previous version of this task (hip2hip/NormalAttention_dot) used 4 channels and
hw up to 4096, which is a toy shape; channels are rescaled to 64 (the width such blocks use) and hw to
1024-16384. Workloads: 8 performance cases span B*hw = 1024 to 131072 (128x); the cost is about
160 * batch * hw^2 flops, so even the smallest case is far above the 10 us floor. Eight small, odd and
minimum shapes (hw=1, 16, 50, 100, 333, 1000, 1025, 2049; batch 1, 3, 5, 17, 2) are correctness-only.
Weights are N(0, 0.0625) (0.125 for the output convolution) and biases N(0, 0.1), so the energies are of order one
and the ELU is exercised on both branches.

## Baseline

A HIP baseline, task-original: the previous task provided a hipBLAS/ATen-convolution composition (not
allowed as a baseline here: no BLAS, no ATen compute, and it materializes the hw x hw energy). This
baseline is a rewrite of it in plain HIP: a fused q/k/v projection kernel and a tiled attention kernel that applies the output convolution in its epilogue, never storing the energy. Provenance is the in-repository hip2hip task at the recorded revision; the kernel
is float32 throughout with fp32 accumulation. Channel counts are compile-time constants (64/16), which
is why the scenario fixes them.

## Reference and bound

`reference.py` evaluates the block in float64, chunking the summed position to bound memory. It was
checked on CPU against the original functional module.

`out` uses `baseline_relative` with `k: 8` and `max_violation_fraction: 0.002` instead of the default
`k: 2` with no violations (a bound override).

The baseline is not more accurate than float32 accumulation. It is plain float32 with FMA chains in
one fixed order: each projection starts from its bias and adds 64 products, the energy is a 16-term
dot product, the summed position i is accumulated sequentially in one register per output (hw terms)
and the output convolution starts from its bias and adds 64 products. Its error is float32 rounding
noise of about 5 ulp RMS (relative to the mean |out|) for hw up to 333. Because the sum is
sequential, the error grows to about 10 ulp at hw = 4096 and 20 ulp at hw = 16384 (SQNR from 132 dB
down to 120 dB); blocked or pairwise sums are more accurate there. Any other correct float32
implementation has independent rounding noise of the same size: another summation order, FMA
contraction, bias position or tiling, or the reference itself evaluated in float32. The default bound
rejects that noise. On MI355X the QA calibration probe (the reference evaluated in float32 PyTorch)
failed 13 of the 16 cases with `k: 2`: 12.5% of the 64 elements of `b1_n1` and up to 1.9% of the
elements of the other cases were out of bound, by up to 3.0 times the bound. `b16_n8192` failed on
the bias test alone.

The new bound comes from a CPU study. It compared a float32 emulation of the baseline's exact
arithmetic (the same FMA order) and 13 independent correct float32 implementations with the
float64 reference, using the evaluator's own comparison. It covered all 16 workload shapes; the five
largest were subsampled to 2 images and 384 output positions, with the bias test applied at the
full case's size. Each shape had 3 to 22 draws per implementation. The implementations were:

- the reference evaluated in float32 PyTorch;
- MFMA-like k=4 chunked sums in every product, with the biases added last;
- no FMA contraction;
- division by hw instead of multiplication by fl(1/hw), and 1/hw applied after the sum;
- the projection bias added after the dot product;
- 4- and 32-way interleaved partial sums, the sequential sum reversed, 256-position blocks, k=4
  chunks and pairwise summation;
- the output convolution folded into v (`out = (Wg v) energy + bg`).

Results at hw >= 16 (all cases except `b1_n1`):

- `k: 2` failed 1628 of 1900 (case, implementation, draw) samples, with up to 4.4% of the elements
  out of bound.
- `k: 4` with 0.001 failed 107 samples (89 elementwise). `k: 8` with no allowance failed 88 (82
  elementwise).
- `k: 8` with 0.002 failed none elementwise. The worst violation fraction was 9.8e-4 (one element of
  1024 at `b1_n16`), and at most 1.6e-4 for hw >= 50. The allowance is twice the worst fraction seen.

The bound still rejects every QA control by a wide margin. A 16-ulp relative shift of the baseline
output is rejected in every draw of every case with hw <= 333. At hw <= 100 the rejection is
elementwise: 1.6% to 39% of the elements at hw = 1, and at least 0.49% at hw = 16, 0.82% at hw = 50
and 0.54% at hw = 100, which is 2.4 to 4.1 times the allowance. In the larger cases the bias test
rejects it in 104 of 114 draws, because its mean error is 2 to 10^6 times the baseline's. Float8
rounding violates more than 99.9% of the elements. Rounding the attention inputs v and energy to
TF32 violates 31% to 100% of them.

Two residual risks come from how the bound is computed, not from this task:

- `b1_n1` (hw = 1, the minimum): every row of the `[batch, 64, hw]` output (one channel across
  positions) has a single element, so the bound's per-row RMS floor is that element's own baseline
  error. An independent implementation fails whenever that error happens to be small. Implementations
  that change the projection or output-convolution arithmetic failed 13 to 17 of 18 draws at `b1_n1`
  even with `k: 8` (up to 11% of the 64 elements). Reorderings of the attention sum, which are
  identical at hw = 1, never failed. No allowed `k` or `max_violation_fraction` changes this. The case
  is kept because it is the minimum of the hw axis.
- In the performance cases, the rounding errors of the q, k and v projections are shared by all
  positions. A correct implementation's mean error is therefore well above 5/sqrt(n) of its RMS, and
  the bias test reduces to comparing the two independent mean errors. With `k: 8` this rejected 10 of
  680 performance-case samples (1.5%) and none of the others. With `k: 4` the bias test alone
  rejected 18 samples.

## GPU audit

The MI355X QA audit of the previous revision (default bound) passed baseline compile, correctness,
CUDA-graph timing of every performance case above the 10 us floor and the full control suite; its
calibration probe is the reason for the bound override above.
