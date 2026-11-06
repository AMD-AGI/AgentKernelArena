# mHC fused post + pre with output RMSNorm, 4 streams, hidden 4096

One layer boundary of a manifold-constrained hyper-connection (mHC) model with 4 residual streams
(DeepSeek-V4-Flash: hidden size 4096, 20 Sinkhorn iterations, every epsilon 1e-6, post multiplier
2). The **post** block of one layer mixes the residual streams and adds the layer output `x`. The
**pre** block of the next layer then projects the new streams to 24 mixes and reduces them to that
layer's input, which an RMSNorm with `norm_weight` normalizes:

```
next_residual[t, j, :] = post_mix[t, j] * x[t, :] + sum_i comb_mix[t, i, j] * residual[t, i, :]    (stored bf16)
F           = flatten(bf16(next_residual))                                   # [tokens, 16384]
mixes       = (F @ proj_weight^T) * rsqrt(mean(F^2) + rms_eps)               # proj_weight: fp32 [24, 16384]
pre_gate    = sigmoid(mixes[:4] * mix_scale[0] + mix_bias[:4]) + pre_eps
next_post_mix = sigmoid(mixes[4:8] * mix_scale[1] + mix_bias[4:8]) * post_multiplier
next_comb_mix = Sinkhorn(softmax(mixes[8:].view(4, 4) * mix_scale[2] + mix_bias[8:]) + sinkhorn_eps)
layer_input = RMSNorm(sum_i pre_gate[i] * bf16(next_residual)[i], norm_weight, norm_eps)      (bf16)
```

The Sinkhorn step divides by the column sums (+ eps), then repeats row and column normalization
`sinkhorn_iters - 1` times, adding the epsilon to every denominator. This is AITER's HIP
exp-domain variant. The operator is AITER's `mhc_fused_post_pre` with `norm_weight`, the op that
vLLM's DeepSeek-V4 model on ROCm runs at every attention and MLP boundary. It replaces this
task's source task, a FlyDSL task whose baseline called that HIP op.

## Interface and scenario

`tokens` varies. The stream count, hidden size, scalars and dtypes are fixed: bfloat16 `x`,
`residual` and `norm_weight`; float32 mixes and projection weights, in the shapes of the serving
trace. Two outputs drop or flatten trivial dimensions relative to production:

- `next_post_mix` is `[tokens, 4]` (production returns the same memory as `(M, 4, 1)`).
- `next_comb_mix` is the row-major `[tokens, 16]` view of the `(M, 4, 4)` matrix, the layout
  the kernel stores.

The flat layout makes the baseline-relative bound use the RMS over each token's 16 entries.
Sinkhorn is invariant to per-row and per-column logit shifts. As a result, the baseline's own error
can vanish in a single 4-entry row, and a CPU emulation then rejected correct implementations, such
as one taking the sum of squares from the rounded residual, on about 6e-6 of the elements. With
16-entry rows every correct variant stays below 0.8 of the bound over 1.3e5 emulated tokens.

## Reference

The reference is float64 PyTorch and independent of the baseline. It follows the source task's
reference, re-derived in float64:

- **bfloat16 residual.** The pre block consumes the bfloat16 `next_residual` the operator stores,
  so an implementation must round the post-mapped residual before the projection. Keeping it in
  float32 fails the mixes. The returned `next_residual` is the unrounded value; the comparison
  rounds it.
- **Un-normalized layer input.** The gated stream sum stays in high precision and only the
  normalized `layer_input` is rounded to bfloat16, as the source task defines it.

AITER's HIP kernel and vLLM's TileLang kernel instead stash the un-normalized layer input in
bfloat16 before scaling it. Emulated on the CPU, that second rounding puts about 0.06% of the
`layer_input` elements out of bound, 60 times the allowance below, and fails every case. A port of
those kernels must keep that intermediate in float32.

## Baseline

The baseline is AITER's Triton `mhc_post_pre`, vendored from ROCm/aiter at the recorded commit (MIT
license, `baseline/LICENSE`):

- **Launch 1** fuses the post mix with the split-K partial projection and its sum of squares.
- **Launch 2** reduces the partials, applies the gates and Sinkhorn, and writes the gated stream
  sum.
- **Launch 3** is AITER's Triton RMSNorm (`_rmsnorm_forward`), because the AITER Triton mHC path
  does not fuse the output norm.

The kernels are byte-identical to upstream. The launcher keeps its tuned gfx950 configuration for
C=4096 and the upstream M-threshold selection, inlined without file I/O.

`run()` makes four choices:

- It passes the float32 `[24, 16384]` weight as a transposed view; the kernel rounds it to
  bfloat16, as upstream does.
- It selects the HIP-compatible exp-domain Sinkhorn (`asymmetric_exp_domain=True`), which matches
  the operator.
- It keeps the un-normalized layer input in a float32 buffer, which the kernel's store casts to
  and the RMSNorm reads.
- It writes the preallocated outputs, `next_comb_mix` through a `(tokens, 4, 4)` view.

The split-K reduction is deterministic. When the target language is Triton, the candidate starts as
this baseline.

## Inputs

The inputs are synthetic, in the source task's style:

- `x` and `residual`: unit normal.
- `post_mix`: uniform in (0.1, 1.9), the range of `2 * sigmoid`; the source task used (0.1, 0.9).
- `comb_mix`: near doubly stochastic, from a softmax of normal logits followed by 20 row/column
  normalizations (`inputs.py`).
- `proj_weight`: fan-in scaled, std 1/128, so the normalized mixes are O(1).
- `mix_scale`: uniform in (0.5, 1.5); the source task used ones.
- `mix_bias`: normal with std 0.1.
- `norm_weight`: uniform in (0.5, 1.5).

The sigmoids and Sinkhorn therefore run away from their symmetric points (post gates in
0.04-1.96, Sinkhorn entries in 3e-4 to 0.96).

## Workloads

The workloads are DeepSeek-V4-Flash token counts:

- decode batches of 1-16 and 64;
- the unaligned 200 and 1000;
- 256 to 32768 for performance;
- the 64- and 7211-token batches of a profiled MI355X serving trace.

Batches below about 256 tokens move less than 30 MB and fall under the 10 µs timing floor, so they
are correctness-only. The performance cases cover every configuration bucket of the tuned table.

## Comparison

`next_comb_mix` and `next_residual` use the default `baseline_relative` bound. Two outputs override
it (QA warnings, justified here):

- `next_post_mix`: `k: 4` and `max_violation_fraction: 1e-4`.
- `layer_input`: `max_violation_fraction: 1e-5`; `k` stays 2.

The evidence is a CPU emulation of the baseline's arithmetic: bfloat16-rounded weights, float32
split-K accumulation, the sum of squares of the unrounded residual and float32 Sinkhorn. It was
run against correct alternatives: another float32 summation order of the post mix, float32 weights
with whole-K accumulation, the sum of squares of the rounded residual, and reciprocal-multiply
Sinkhorn with `exp2`, alone and combined. Under the default bound, these correct alternatives fail
on isolated elements, for two reasons:

- **bfloat16 rounding flips of the residual.** The reference rounds the exact post mix to bfloat16.
  A float32 implementation rounds its own float32 value, so about one residual element in 3e4 lands
  on the other side of a rounding boundary. That one-ulp change moves a `layer_input` element by up
  to about one bfloat16 ulp. When the baseline's own error there is small, the default bound
  rejects it. That happened to about one `layer_input` element in 1e8 to 2e8; the cases hold 3e8
  elements per draw. A one-draw emulation of the whole suite rejected the other summation order on
  one `t8192` element.
- **Four-entry rows.** The per-row RMS of `next_post_mix` covers only 4 entries. About once in 1e6
  elements all four baseline errors are small together, and a more accurate candidate exceeds twice
  that error. Over 1.3e5 emulated tokens the peak was 1.15 times the default bound.

The allowances sit far above these rates but below any structured error. In the 32768-token case,
1e-5 of `layer_input` is 1342 elements, a third of one token's row, and 1e-4 of `next_post_mix` is
13 entries. The smallest cases allow no violation, and they hold too few elements for the rare
events to matter. With these bounds the one-draw suite emulation passes every correct alternative
above, and the wrong implementations still fail:

- the bfloat16 layer-input stash, in every case;
- a float32 residual fed to the projection, from 3 tokens up, mostly on the mixes;
- an untransposed `comb_mix`, by hundreds of times the bound.

The baseline's SQNR against the reference is about 58-68 dB for the mixes, 63-65 dB for the layer
input and about 100 dB for the residual.

## Risks for the GPU audit

- **Timing floor.** The 256-token case is estimated at about 12 µs (37 MB moved plus three
  launches). If it measures below 10 µs it must become correctness-only, and the performance span
  (now 128x) then needs a 65536-token case.
- **Memory.** The 32768-token case needs about 14 GB for its float64 reference.
- **Untested on GPU.** Nothing has run on a GPU yet. The CPU pre-check did not smoke the
  32768-token reference, which is too large for the CPU host; it ran 7211 tokens instead.
