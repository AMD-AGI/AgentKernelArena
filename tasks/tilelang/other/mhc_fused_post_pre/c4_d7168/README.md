# mHC fused post + pre block, 4 streams, hidden 7168

One layer boundary of a manifold-constrained hyper-connection (mHC) model with 4 residual streams
(DeepSeek-V4 style): the **post** block of layer L mixes the residual streams and adds the layer
output `x`, then the **pre** block of layer L+1 reduces the new streams to a layer input and produces
the next post/combination mixes.

```
residual_cur[t, j, :] = sum_i comb_res_mix[t, i, j] * residual[t, i, :] + post_layer_mix[t, j] * x[t, :]   (bf16)
mixes      = (residual_cur @ fn^T) * rsqrt(mean(residual_cur^2) + rms_eps)          # fn: [24, 4 * 7168]
pre_mix    = sigmoid(mixes[:4] * hc_scale[0] + hc_base[:4]) + hc_pre_eps
post_mix   = sigmoid(mixes[4:8] * hc_scale[1] + hc_base[4:8]) * hc_post_mult_value
comb_mix   = Sinkhorn(softmax(mixes[8:].view(4, 4) * hc_scale[2] + hc_base[8:]) + eps)  # 20 iterations
layer_input = sum_j pre_mix[j] * residual_cur[t, j, :]                                # bf16
```

No output RMSNorm is fused (the `norm_weight=None` form of vLLM's op). Hidden size, stream count,
epsilons and the 20 Sinkhorn iterations are the DeepSeek-V4-Pro values of the profiled serving
traces; `tokens` varies.

**Reference.** Float64 and independent of the baseline. The pre block consumes the bfloat16
`residual_cur` that the operator stores (the composition of vLLM's unfused `mhc_post` and `mhc_pre`),
so a candidate must round the post-mapped residual before the projection; the returned
`residual_cur` is the unrounded value (the comparison rounds it to bfloat16). The baseline's
small-batch path (at most 16 tokens) keeps the unrounded residual inside its fused projection, so
its own error against this reference is larger there and the baseline-relative bound is looser for
those correctness-only cases.

**Baseline.** vLLM's TileLang implementation (`mhc_fused_post_pre_tilelang` and its kernels,
vllm-project/vllm at the recorded revision, Apache-2.0), specialised to what runs on ROCm: no
DeepGEMM (so the TileLang prenorm-GEMM kernels and their size-based dispatch are used, split-K 1),
no PDL, and no output norm. Dispatch: up to 16 tokens a single fused post + projection kernel;
otherwise the post kernel, then a prenorm GEMM (a 1024-thread kernel below 128 tokens, a block-M
kernel from 1024 tokens, the default kernel in between), then the big fused pre kernel (mixes,
Sinkhorn on one wave, layer input). The kernels are unchanged apart from removing the CUDA-only PDL
hooks and platform imports, and from the reduction change below. Reduction change for AMD wavefronts: upstream's three projection kernels
(`mhc_fused_tilelang`, `hc_prenorm_gemm_tilelang`, `hc_prenorm_gemm_block_m_tilelang`) finish their
per-thread partial sums with `T.warp_reduce_sum` and a 32-lane warp split (`lane = tid % 32`, one
shared slot per 32 threads). The first MI355X audit run of the unmodified copy showed the signature
of that assumption failing on 64-lane wavefronts: the 1024-token and larger cases (block-M and default
kernels) produced finite outputs whose `post_mix`, `comb_mix` and `layer_input` SQNR against the
float64 reference was only 23-39 dB while `residual_cur` (not reduced) was at 96-98 dB, and the
smaller cases returned nonfinite mixes. A CPU emulation with the projection scaled by sqrt(2) (each
wave's partial sum and square sum counted twice, which is what a 64-lane warp reduction followed by
two 32-lane writes gives) reproduces that error level (about 30-34 dB for the same three outputs),
so the vendored copy replaces that tail: every thread stores its partials to shared memory, 32-element
groups are summed serially by the first `n_thr / 32` threads, and the first `tile_n + 1` threads (or
`block_m * (tile_n + 1)` for the block-M kernel) sum the groups and store the results. It computes the
same sums, with the same dispatch, launch grids and per-thread accumulation, independent of the
hardware wave size. The round-3 MI355X QA audit ran this copy on every case and passed.
One deviation: upstream passes `n_splits=` to `mhc_fused_tilelang`, a
name that kernel does not have (its split count is `split_k`); the vendored call passes `split_k`.
The baseline is a TileLang baseline, so candidates in every target language start from an
unimplemented template and there is no same-language positive control for it (the audit's
QA-POSITIVE-UNAVAILABLE warning). It is kept because it is the production kernel: vLLM implements
the fused operator only in TileLang on CUDA and ROCm (`MHCFusedPostPreOp.forward_hip` dispatches to
`mhc_fused_post_pre_tilelang` and otherwise falls back to the unfused PyTorch `mhc_post` and
`mhc_pre`), so a Triton baseline would replace the production kernel with task-authored code.

**Workloads.** Token counts: decode batches 1-16 (single fused kernel), 17 and unaligned 1000
(correctness only), and 64 to 16384 for performance, including the 64 and 7211 token shapes of the
profiled serving traces. Inputs are synthetic: unit-normal streams, `fn` with std 0.001 (so the
projected mixes are O(0.1) after normalisation), `post_layer_mix` in (0, 2), row-stochastic
`comb_res_mix`, `hc_scale` in (0.5, 1.5) and `hc_base` with std 0.5 so that the sigmoids and the
Sinkhorn iteration are exercised away from their symmetric point.

**Comparison.** `residual_cur` keeps the default bound (`baseline_relative`, k 2). The three outputs
computed from the projection use calibrated `baseline_relative` bounds: `post_mix_cur` and
`comb_mix_cur` `k: 8` with `max_violation_fraction: 0.001`; `layer_input_cur` the default k with
`max_violation_fraction: 1.0e-5`.

*Why the default rejects a correct float32 implementation.* Any float32 implementation computes the
post-mapped residual in float32 and rounds it to bfloat16 before the projection, as the operator
stores it. Where the exact value lies within float32 rounding of a bfloat16 rounding midpoint, the
stored element differs by one bfloat16 ulp from the float64 reference's rounding. In a CPU emulation
this happens to about one of the 28672 elements of a token (0.75 to 0.95 per token for different
operation orders), in the baseline and in any other float32 design alike, but at different elements.
Each such element moves a projection by `fn[o, k]` times its ulp; removing all float32 accumulation
error from the emulated projection changes the `post_mix_cur` error RMS by under 0.5%, so the mix
errors of every float32 design are a few such events per token, a heavy-tailed distribution
(kurtosis about 50 for the float32 evaluation's `post_mix_cur`). `post_mix_cur` and `comb_mix_cur`
rows have 1 and 4 elements, so the comparator scales them with the whole output's RMS of the
baseline error, and a token whose events fall on large elements exceeds twice that RMS. The round-3
MI355X QA probe (a
float32 PyTorch evaluation of the reference, one draw per case) failed the default bound in seven
cases for each output (violation fraction / peak bound multiple): `post_mix_cur` t1000 0.00075 /
1.15x, t256 0.00098 / 1.08x, t1024 0.00098 / 1.34x, t2048 0.00146 / 1.92x, t4096 0.00031 / 1.53x,
t7211 0.00045 / 2.41x, t16384 0.00056 / 2.89x; `comb_mix_cur` t1000 0.0005 / 1.36x, t64 0.00098 /
1.55x, t1024 0.00061 / 1.62x, t2048 0.00116 / 2.27x, t4096 0.00055 / 2.36x, t7211 0.00058 / 2.51x,
t16384 0.00074 / 2.87x; bias z at most 0.031. In `layer_input_cur` (bfloat16, per-row RMS) the same
events occasionally push an element whose magnitude is large for its row across a bfloat16 midpoint,
a one-ulp error just over the bound: one element in t2048 (6.8e-8 / 1.03x) and one in t16384
(8.5e-9 / 1.04x), bias z at most 1.1e-4.

*The baseline is not more accurate than float32.* Its prenorm GEMM kernels (upstream code) compute
`x_val * x_val` on two bfloat16 loads, so the square sum accumulates bfloat16-rounded squares: a
relative error of about 1e-5 in the RMS factor shared by all 24 mixes, which makes the baseline's
mixes about 14 dB less accurate than the float32 evaluation's. A CPU emulation of the baseline with
that rounding reproduces the probe at t1024, t2048 and t4096 (violation fractions 1.5e-4 to 6.7e-4,
peaks 1.5x to 2.5x; the GPU probe 3.1e-4 to 1.5e-3, 1.3x to 2.4x), while float32 squares give
20 to 80 times more violations (peaks 6.7x to 11.4x), and the emulated `residual_cur` SQNR (about
97 dB) matches the first MI355X audit's. The probe failures therefore come from the heavy tail, not
from a more accurate baseline.

*Calibration.* For these float32 outputs the rounding term `u|y|` is under 3% of the bound, so `k: 8`
widens the bound by a factor of at least 3.9 over k 2: every probe row's peak falls to at most about
0.74x and no element violates. The CPU emulation (the float32 evaluation and a second float32 design
with another operation order, against the emulated baseline; 4 draws of every case plus 300 draws of
t17 and t64, 40 of t256 and 12 of t1000) has no violation at `k: 8` except single elements of
`comb_mix_cur` in t16384 (3.8e-6, peak at most 1.08x, in 2 of 4 draws). Where the bias test needs
its baseline fallback (`comb_mix_cur` in some draws from t1000 up; t4096, t7211 and t16384 in the
probe), the emulated mean error uses at most 0.12 of the allowance at `k: 8` (0.43 at k 2).
`max_violation_fraction: 0.001` admits such a rare token (a single element is at most half of it
from t1000 up for `post_mix_cur` and from t256 up for `comb_mix_cur`), and every admitted element
must still lie within 16x its bound. `k: 6` would leave the probe's peaks at about 0.97x and still
gave violations in the small cases (1 of 300 emulated draws of t17, 2 of 300 of t64, 1 of 40 of t256,
1 of 12 of t1000), where one element is 0.006% to 1.5% of the output. For `layer_input_cur` a larger
k does not remove the one-ulp events (one at t16384 still exceeds k 4), so k stays 2 and
`max_violation_fraction: 1.0e-5` admits one of them in any case from t17 up, with half the allowance
to spare from t64 up (emulated: at most 2 per case, violation fraction at most 1.4e-7, peak at most
1.45x; the probe's 6.8e-8 is under half of it).

*What the bounds still reject.* In the first case (t1), the 16-ulp shift and fp8 rounding controls fail
`residual_cur` under its unchanged default bound (emulated violation fractions 0.95 and 0.68-0.69,
peaks 13x and 11x) and `layer_input_cur` (0.95 and 0.68-0.69); fp8 rounding also fails `post_mix_cur`
and `comb_mix_cur` at `k: 8` (0.75 to 1.0, peaks 27x to 82x). A 16-ulp float32 shift of the mixes
is below the baseline's own error and is caught there only by the bias test, in some of the larger
cases. The noop control leaves the NaN poison; zeros, half_written and stale_outputs give errors
of the order of the values themselves in at least half of every output, far beyond 16x the bound
and the allowed fraction; input_write is caught by the input-mutation check whatever the bound.
Semantic errors stay rejected in every case from t17 up: a projection over the unrounded float32
residual (the numerics of the at most 16 token path) or over bfloat16-rounded `fn` fails
`post_mix_cur` and `comb_mix_cur` at `k: 8` with violation fractions of 0.57 to 0.90 and peaks of
7.5x to 20x in every emulated draw.

**Risks for the GPU audit.** The 64-token case moves about 17 MB (4 us at memory speed) and relies
on four launches to clear the 10 us floor; the round-3 audit measured it above the floor, but if a
later run measures it below, it must be demoted to correctness-only. The 16384-token case needs
about 15 GB for inputs plus the float64 reference. The calibrated bounds rest on one probe draw per
case and on a CPU emulation of the baseline's numerics; the next audit's probe is their GPU check.
