# Transformer FFN with a 1x1 convolution and GELU, float32, hidden 512 -> 2048 -> 512

`out = gelu(x @ conv_weight^T + conv_bias) @ linear_weight^T + linear_bias`: the feed-forward layer of a
TransformerFFNLayer whose first layer is a `Conv1d` with kernel size 1 and SAME padding, followed by the exact
(erf) GELU and a `Linear`. With kernel size 1 the convolution is a per-token linear map and its
`kernel_size ** -0.5` output scale is 1, so the conv weight is declared as the `[filter, hidden]` matrix
(the trailing unit kernel axis dropped). The time and batch axes of the old `(T, B, C)` input are one token
axis `n_tokens = T * B`, since no layer mixes tokens.

- **Merged sources.** `hip2hip/TransformerFFNLayer` and `torch2hip/gpumode/14069_TransformerFFNLayer` are the
  same operator (gpumode 14069) with `hidden_size=512`, `filter_size=2048`. The incremental-state, LEFT-padding,
  ReLU and dropout paths of the original module are unused by the workload and are not part of the task.
- **Inputs.** The weights and biases use PyTorch's default `Conv1d`/`Linear` initialization, uniform within
  `+-1/sqrt(fan_in)`; the activations are standard normal.
- **Baseline.** Task-original HIP. The provided HIP source of the old tasks delegated both GEMMs to ATen
  (`conv1d`, `matmul`) and only fused the GELU and the bias add, which the HIP candidate rules forbid, so the
  baseline is a handwritten pair of tiled float32 GEMM kernels (64x64 output tile, 16-deep k step, 4x4 outputs
  per thread, no matrix instructions): the first fuses bias and exact GELU, the second adds the second bias. The
  intermediate `[n_tokens, 2048]` buffer is allocated inside `run()`. It is deliberately not a rocBLAS-class
  GEMM; a candidate can beat it by a wide margin.
- **Workloads.** Token counts 1024 to 8192 are the gpumode cases (`T x B` = 128x8, 256x8, 512x8, 512x16);
  128, 256, 512 and 16384 extend the range to the 100x span. The smallest performance case still does
  0.5 GFLOP in two GEMMs, so the baseline runs far above the 10 us floor. Token counts 1, 3, 48 and 1000 are
  correctness-only boundary cases. The largest case holds a 268 MB float64 intermediate in the reference.
- **Comparison.** `baseline_relative` with `k: 4` and `max_violation_fraction: 0.01`, calibrated below against
  the MI355X audit, a CPU emulation of the baseline's arithmetic and the error-bound controls.
- **Risk.** A float32 GEMM written with plain FMAs is compute-bound, so the 16384-token case is the slowest. The
  `shifted_16_ulp` control is rejected by the bias test, which depends on the mean of each input draw: the rejecting
  case on the GPU's draws is not known in advance (see Comparison).

## Comparison

The baseline is a plain sequential-FMA float32 kernel. On MI355X it measures 120.4 to 120.5 dB SQNR against the
float64 reference in every case, an error RMS of about 16 float32 unit roundoffs relative to the output RMS from the
512- and 2048-term sequential dot products. The earlier `sqnr` floor of 123 dB had been calibrated on the CPU against
MKL's float32 GEMM (126 to 129 dB, blocked accumulation) and rejected the baseline itself. A CPU emulation of the
kernel's arithmetic (per output one sequential float32 FMA chain in the kernel's `k` order for each GEMM, float32 bias
adds, the exact-erf GELU in float32 operations) with the task's inputs and case seeds gives 120.4 to 120.6 dB from 3
tokens up (119.6 and 120.1 dB for the two 512-element single-token draws), within 0.1 dB of the MI355X measurement. The
`shifted_16_ulp` control adds a relative error of 16 unit roundoffs, which alone is 120.4 dB, the size of the baseline's
own error: the shifted baseline measures 116.9 to 117.7 dB, only 3 dB under the baseline and under a correct float32
design that accumulates in another order (120.5 dB, more spread on the single-token case). The bound therefore returns
to the default per-element `baseline_relative` with its bias test.

**Calibration.** The QA audit's float32 PyTorch evaluation of the reference on MI355X, under the default `k: 2`,
exceeded the bound in 10 of 12 cases. That is float32 accumulation noise of the baseline's own size: the emulated
kernel with both `k` loops reversed (a correct float32 design whose rounding is independent of the baseline's) exceeds
`k: 2` on 3.4% to 3.6% of the elements, peak 3.2 to 5.9 times the bound. Read along the reversed kernel's
violation-fraction curve, the GPU evaluation behaves like it with 0.94 to 0.96 times its error from 512 tokens up and
0.42 to 0.53 times at 48 to 256 tokens. The projection to `k: 4` scales each case along that curve, and the peak by the
reversed kernel's peak ratio between `k: 4` and `k: 2`:

| Case | Audit float32, `k: 2` (fraction, peak) | Projected, `k: 4` (fraction, peak) | Reversed-order kernel, `k: 4`, worst of 2 draws |
| --- | --- | --- | --- |
| n1 | 0, 0.54 | 0, 0.27 | 0.0020 (1 of 512 elements), 1.13 |
| n3 | 0, 0.78 | 0, 0.40 | 0.0026 (4 of 1536), 1.12 |
| n48 | 0.00037, 1.26 | 0.00002, 0.66 | 0.0015, 2.22 |
| n128 | 0.0022, 1.63 | 0.00001, 0.83 | 0.0016, 1.75 |
| n256 | 0.0022, 1.78 | 0.00001, 0.93 | 0.0017, 2.38 |
| n512 | 0.0288, 4.00 | 0.0011, 2.08 | 0.0016, 2.26 |
| n1000 | 0.0277, 5.66 | 0.0010, 2.97 | 0.0016, 2.30 |
| n1024 | 0.0289, 4.14 | 0.0012, 2.10 | 0.0016, 2.38 |
| n2048 | 0.0292, 4.01 | 0.0012, 2.09 | 0.0015, 2.33 |
| n4096 | 0.0297, 4.21 | 0.0011, 2.19 | 0.0015, 2.32 |
| n8192 | 0.0294, 4.75 | 0.0011, 2.49 | 0.0015, 3.12 |
| n16384 | 0.0290, 4.92 | 0.0011, 2.57 | 0.0015, 3.00 |

At `k: 4` the float32 evaluation is projected to exceed the bound on at most 0.12% of the elements with a peak of at
most 3.0 times the bound, and the reversed kernel measures at most 0.26%, peak 3.1: both under half of the 1% allowance
and far below the 16-times peak limit. The audit's bias z stayed at most 0.54 of the `5 / sqrt(n)` threshold, and the
bias test does not tighten with `k`. At `k: 3` the projection is 0.56%, above half of the allowance, and at `k: 3.5`
it is 0.25%, exactly half, so 4 is the smallest `k` with margin. Under the committed comparator with this declaration,
the emulated baseline copy, the reversed kernel, MKL float32 and a split-K (128-deep chunks) float32 evaluation pass all
24 checks (12 cases, 2 seeds each); MKL and split-K float32 are 6 to 13 dB more accurate than the baseline and have no
violations.

**Controls.** No `k` and `max_violation_fraction` reject `shifted_16_ulp` elementwise while the float32 evaluation stays
within half of the allowance. Up to `k: 2.75` the float32 evaluation's projected fraction (1.3% at 2.5, 0.84% at 2.75)
would need an allowance above the 1% cap. From `k: 3` the shift's largest fraction (0.41% at `k: 3`, 0.012% at `k: 4`)
lies below the float32 evaluation's (0.56% at `k: 3`). The bias test rejects it instead. The shift adds
16 * 2^-24 * mean(y) to the mean error, while the baseline's own mean error is unbiased (within 2 standard errors in
every emulated check). Because the output's column means do not cancel, mean(y)/rms(y) is about 0.015 RMS over input
draws. With the task's case seeds on the CPU at `k: 4`, the shifted output fails both the z test and compare.py's
fallback `|mean(e)| <= k * |mean(e_b)| + 5 * rms(e) / sqrt(n)` in five checks:

| Check | mean(y)/rms(y) | z over `5 / sqrt(n)` | \|mean(e)\| over the fallback limit |
| --- | --- | --- | --- |
| n256, second seed | +0.037 | 2.05 | 1.36 |
| n4096, second seed | +0.010 | 2.36 | 1.22 |
| n8192, first seed | +0.027 | 7.61 | 3.66 |
| n8192, second seed | -0.012 | 3.57 | 2.73 |
| n16384, first seed | -0.009 | 3.66 | 2.77 |

The GPU draws come from the CUDA generator and differ from these, so the rejecting case can move. Over 30 to 60 fresh
draws per case, the shift is rejected in 87% of the n16384 draws, 77% of n8192, 67% of n4096, 46% of n2048 and about a
third of n1000 and n1024. The chance that it passes all 24 checks of an audit is about 4e-6. `fp8_rounded` is rejected
elementwise in every check: at least 99.8% of the elements exceed the bound, with a peak above 17,000 times the bound.
Zeroed, half-written, stale, no-op and input-writing outputs fail at any `k`.
