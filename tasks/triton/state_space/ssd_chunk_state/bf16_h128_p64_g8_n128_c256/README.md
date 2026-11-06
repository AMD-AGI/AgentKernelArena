# Mamba2 SSD chunk state, bfloat16, 128 heads x 64 channels, 8 groups, state 128, chunk 256

State-accumulation stage of the chunked ("state space duality") Mamba2 prefill. For every chunk of 256 tokens
and head, the outer products of the chunk's inputs and B vectors are summed with a decay towards the chunk end:

```text
states[c, h, p, n] = sum_k x[k, h, p] * B[k, g, n] * exp(dA[255] - dA[k]) * dt[k]      (g = h // 16)
```

`dA` is the chunk-local cumulative decay (an input here, produced by the chunk cumsum stage) and `states` is
float32. The sum is a `[64 x 256] @ [256 x 128]` product per chunk and head.

- **Origin.** vLLM's `_chunk_state_fwd_kernel`
  (`vllm/model_executor/layers/mamba/ops/ssd_chunk_state.py`, derived from `mamba_ssm`). SGLang carries a copy
  of the same kernel (the earlier AgentKernelArena Triton-to-FlyDSL task), so both source tasks map onto this one
  scenario. The shape follows hybrid Mamba2 models such as Nemotron-H; the earlier tasks used a few tiny shapes.
- **Baseline.** The vLLM Triton kernel unchanged apart from inlining `fast_exp`. The upstream autotune (it tunes
  on the first call, which is not capturable and keys on constants only) is replaced by one fixed configuration
  (tile 64 x 64 x 32, 4 warps, 2 stages); the launch is wrapped in `run()`. The fixed configuration was chosen
  without GPU timing; it is valid but not necessarily the fastest upstream configuration.
- **Inputs.** `x` and `B` are standard normal bfloat16. `dt` is U(0, 0.1) and `dA_cumsum` its running sum with a
  per-head negative rate in (0.02, 0.2) over the chunk's valid tokens, computed by `inputs.py` (constant
  `dA_cumsum` in the padding of a partial chunk, as the chunk cumsum stage produces). The padding slots of `dt`
  keep their U(0, 0.1) draw instead of the cumsum stage's zeros: the operator sums over the chunk's tokens only
  (the baseline masks those loads), so a correct kernel never uses them, a candidate that reads past the end of
  a partial chunk is no longer hidden by a zero `dt`, and the 1-token case is not 255/256 exact zeros (which the
  QA input audit rejects as a degenerate draw). The valid-token `dt` and all of `dA_cumsum` are bitwise the same
  as with zero padding, so the reference and the baseline's output are unchanged. `cu_chunk_seqlens` are
  consecutive 256-token chunks.
- **Workloads.** 1024 to 131072 tokens (128x span; about 35 KB of traffic per token). The MI355X QA audit timed
  the baseline at 19.3 us (1024 tokens) to 1482 us (131072 tokens). Smaller counts are correctness-only: 1 token
  (axis minimum), 100 (one partial chunk), 1000 (non-power-of-two, not a multiple of 16) and 2051.
- **Comparison.** `baseline_relative` against a float64 reference with the default `k: 2` and
  `max_violation_fraction: 4.0e-5` (an override of the default 0, justified below). The baseline rounds the
  decay-weighted B operand `B * exp(dA[255] - dA[k]) * dt[k]` to bfloat16 before the dot (as the production
  kernel does), so its own error, and hence the bound, is at the bfloat16 level (SQNR 55.6 dB).
- **QA status.** The auditor's `shifted_16_ulp` control is not rejected, and no change inside the task can
  reject it while accepting correct kernels (see the last section). The task waits for a framework decision on
  that control.
- **Scope.** The earlier tasks also exercised float16 and float32 inputs and a batch axis; bfloat16 with the
  packed-token layout (`cu_chunk_seqlens`) is what hybrid Mamba2 serving uses, so this scenario fixes it. A
  float16 or float32 variant would be a sibling scenario.

## Bound override: `max_violation_fraction: 4.0e-5`

A candidate that keeps the baseline's algorithm (the same bfloat16 rounding of `B * scale`, exact bf16 products,
float32 accumulation) but evaluates the float32 decay scale differently gets a few `B * scale` values on the
other side of a bfloat16 rounding midpoint. Each such flip moves one product by one bfloat16 ulp, and where the
product is large and the baseline's own error small, the element leaves twice the baseline's per-row RMS error.
A different accumulation order alone never does (16- and 64-token steps and sequential accumulation stay at
0.5 times the bound).

The numbers below come from a CPU emulation of the baseline arithmetic: decay `exp2(log2(e) * x)`,
`B * (decay * dt)` rounded to bfloat16, exact products summed in 16-token steps into a float32 accumulator.
Inputs are drawn as the evaluator and `inputs.py` draw them. Every candidate is scored with the evaluator's
current `compare_tensor`; `states` is dense, so every row has 128 compared elements. The emulated baseline has
an SQNR of 55.5 to 55.8 dB. The emulated float32 calibration probe peaks at 8.3e-5 (1 token) to 8.7e-4
(131072 tokens) of the bound; the MI355X audit's probe peaked at 1.1e-4 to 1.4e-3.

Draws per case: 1 token 10, 100 tokens 300 to 350, 1000 and 2051 tokens 8 each, 1024 to 4096 tokens 3 each,
8192 to 65536 tokens 1 each (in 4096-token slices), and 131072 tokens 1 (probe and controls only). The
100-token case (one partial chunk of 1,048,576 elements) is the worst case for every variant.

| Variant of the baseline arithmetic | draws | failing the default bound | worst violation fraction, 100 tokens | worst elsewhere | worst peak |
| --- | --- | --- | --- | --- | --- |
| decay with `exp` instead of `exp2(log2(e) * x)` | 388 | 46 | 8.6e-6 | 7.2e-7 | 1.8x |
| the same plus up to 1 ulp per decay (a hardware `exp2` against a 1-ulp `expf`) | 300 | 187 | 1.43e-5 | (100 tokens only) | 2.0x |
| `(B * decay) * dt` instead of `B * (decay * dt)` | 385 | 144 | 9.5e-6 | 2.6e-6 | 1.9x |
| decay from pre-scaled cumsums, `exp2(dA[255] * log2(e) - dA[k] * log2(e))` | 385 | 71 | 9.5e-6 | 1.9e-6 | 2.2x |
| split decay `exp(dA[255]) * exp(-dA[k])` (or its `exp2` form) | 312 | 112 (118) | 1.05e-5 | 2.4e-6 | 2.2x |
| decay moved by up to 1 / 2 / 4 float32 ulp per token (another `exp2`) | 385-389 | 220 / 295 / 344 | 1.43e-5 / 1.91e-5 / 2.96e-5 | 2.9e-6 / 3.8e-6 / 7.4e-6 | 2.0-2.2x |
| float32 `B * scale` operand (no bfloat16 rounding) | 45 | 0 | 0 | 0 | 3.6e-4x |
| control: `shifted_16_ulp` | 90 | 0 | 0 | 0 | 0.50x |
| control: output rounded to bfloat16 | 45 | 45 | at least 6.4e-2 | at least 7.7e-2 | 3.6-13.7x |
| control: output rounded to fp8 e4m3 | 20 | 20 | at least 0.76 | | 61-3400x |

Every same-precision variant fails the default bound in 12 % (`exp`) to 88 % (4-ulp model) of the draws, and the
realistic ones (a different `exp` with up to 1 ulp of noise, operation order, pre-scaled or split decays) put at
most 1.43e-5 of the elements out of bounds (15 of the 100-token case's 1,048,576). The pessimistic 4-ulp model
reaches 2.96e-5 (31 elements) and exceeds 2.0e-5, the value an earlier emulation with 3 draws per case chose, in
4 of 350 draws. 4.0e-5 (41 elements of the 100-token case) accepts every emulated draw: 2.8x above the realistic
variants, 2.1x above the 2-ulp model and 1.35x above the worst 4-ulp draw. The admitted elements peak at 2.2x
the bound, far under the 16x admission limit. The bound stays more than three orders of magnitude below the
bfloat16 and fp8 output controls. The noop, zeros, half-written, stale and input-write controls fail at any
fraction (non-finite, zeroed or stale outputs, or a modified input). `k` stays at 2, so every other element
keeps the default bound.

## QA control `shifted_16_ulp`

The auditor's `shifted_16_ulp` control scales the baseline's output by `1 + 16 * 2^-24`, a relative change of
9.5e-7. This task's check accepts it, and no bound, input or case inside the task can reject it while
accepting correct kernels. All numbers below are from the emulation above, under the current comparator.

- **Element bound.** An element's bound is `2 * max(|baseline error|, row RMS)` plus one float32 rounding of
  the reference; a row is the 128 state elements of one (chunk, head, channel). Every term
  `x[k] * bf16(B[k] * scale[k])` of the sum carries the bfloat16 rounding of its B operand (relative error up
  to 2^-9), so the baseline's error is proportional to the output. In every emulated row (390 draws, 1 to
  131072 tokens) the baseline's row RMS is at least 840 float32 ulps of the row's largest `|states|` (median
  about 10^4). The shift leaves the bound only where that RMS is below roughly 7.5 to 15 ulps.
- **SQNR.** The control's SQNR equals the baseline's 55.6 dB.
- **Bias test.** With standard normal `x` and `B` the states have zero mean, so the shift barely moves the mean
  error. With post-SiLU `x` and `B` (positive means; 1 to 4096 tokens), the baseline's own mean error is -4.1e-6
  to +3.1e-6 of the mean state from 100 tokens up, mostly negative (round-to-nearest of the bfloat16 operand
  rounds a decreasing value density towards zero). The control's mean error is -0.6 to 1.3 times the baseline's,
  inside the test's allowance of `k` (2) times the baseline's. A float32-operand kernel (mean error near 0) lies
  on the other side of the control.

**Why the varlen sibling rejects it.** `ssd_chunk_state_varlen` (same scenario) has a correctness case of 128
one-token sequences. There each output row (sequence, head, channel p) is
`x[t, h, p] * bf16(B[t] * dt) + S * exp(...)`: one bfloat16-rounded outer-product term plus a carried float32
state `S` (standard deviation 0.5) that is not rounded to bfloat16. Where `|x[t, h, p] * dt|` is small, the
bfloat16 rounding error of the outer product (at most `2^-9 * |x * dt * B|`) shrinks to a few float32 ulps of
`S`, so the baseline's error in that whole row is float32-level. Emulated under the current
comparator (3 draws), the shift violates 6.2e-3 to 6.6e-3 of that case's elements at peaks of 13x to 14x the
bound, in 2.4 to 2.6 % of its rows. Those rows have a median `|x * dt|` of 2.3e-4 (2.6e-2 over all rows; 99 %
of them are below 1e-3) and a baseline row RMS of 4.4 float32 ulps of the row's largest `|states|` (median;
about 510 over all rows).

**Why that does not carry over.** This operator has no term outside the bfloat16 rounding: no carried, initial
or other float32 state is added. Shrinking `x` or `dt` shrinks the output by the same factor, so every row's
relative error stays at the bfloat16 level. A row would be exact only if `bf16(B[k, n] * scale[k])` were exact
for every state index n and every token k of the chunk. That requires `scale[k] = exp(dA[255] - dA[k]) * dt[k]`
to be a power of two. The decay is exactly 1 only for the chunk's last token, so the row also needs a one-token
chunk (1 token, or 256m + 1 tokens) whose `dt` is a power of two. The emulation confirms the construction: with
`dt` drawn from {2^-4, ..., 2^-7}, the baseline is exact in every element of the 1-token case and of the
one-token chunk of a 257-token case. In both cases every tested variant passes (in the 1-token case each
matches the reference exactly) and the shift fails at 16.9x the bound; a 3-token chunk is no longer exact. But
in production `dt` is a float32 softplus output (vLLM's Mamba2 mixer has the chunk cumsum kernel write
`clamp(softplus(dt + dt_bias), 0, inf)` in float32) and practically never a power of two. A power-of-two `dt` is
also what the QA input audit rejects (100 % of the values exact in bfloat16). Such a case would exist only to
defeat the control, so it is not added. The applicability of `shifted_16_ulp` to an output whose production
algorithm is bfloat16-accurate is left to the framework maintainers.
