# Fused MoE, MXFP4 activations and weights, 65 experts, top-9, hidden 6144, intermediate 2048

One routed mixture-of-experts layer in MXFP4 (OCP E2M1 codes with one E8M0 scale per 32 values),
bfloat16 in and out: every token is sent to its 9 selected experts, each expert computes
`SiLU(x W_gate^T) * (x W_up^T)` followed by the down projection `W2`, and the expert outputs are added
with the routing weights. Both GEMMs run on MXFP4 operands: the bfloat16 activations are quantized to
MXFP4 inside the operator, and so is the intermediate before the down projection. This is AITER's
`fused_moe` with `QuantType.per_1x32`, fp4x2 activations and weights, SiLU and
`doweight_stage1 = False` (routing weights applied in stage 2), the call of the source task.

## Scenario

- `hidden_size` 6144, `intermediate_size` 2048 (W1 holds `[gate | up]`, 4096 rows), 65 experts, top-9.
  This is the source task's shape (`fused_moe_per_1x32_d6144_e65_topk9_n4096_k3072_i1024`), the per-rank
  structure of a GLM-5-style MXFP4 MoE layer (hidden 6144, expert intermediate 2048, top-8 routed experts
  plus a fused shared expert) under 4-way expert parallelism: 64 local routed experts plus the shared one.
  As in the source task, the router draws all nine slots over the 65 experts. The source task was named
  `i1024` after the packed byte width of a W2 row (1024 bytes = 2048 values); the logical intermediate
  size is 2048, so this task is named `i2048`.
- Var axis: `num_tokens`, 1 to 4096 tokens (decode batches through prefill chunks), the source task's
  power-of-two sweep. Every case reads the weights of all experts it selects (about 180 MB for one token,
  most of the 1.3 GB from 16 tokens on, estimated at 40 us and 290 us of weight traffic), so every case
  including one token is above the 10 us timing floor; 1, 4, 16, 64, 256, 1024, 2048 and 4096 tokens are
  timed (4096x), 3, 37 and 300 tokens are correctness-only boundary cases, and 128 and 512 tokens from the
  source sweep are correctness-only too. The source sweep's 2, 8 and 32 tokens are left out: 1, 3, 4, 16
  and 37 tokens already exercise the same small-batch paths (the shortened tile grid below 64 tokens and
  the small-M configuration of the MXFP4 quantizer up to 32 rows), and every further tiny case adds to
  the chance, described under Comparison, that a correct candidate with another SiLU rounding fails.
  The layer does about 76 MFLOP per (token, slot); the float64 reference of the 4096-token case does
  2.8 TFLOP, so larger prefill chunks are left out.

## Semantics (the reference)

For a token `x` routed to expert `e` in slot `k` with weight `w_k`:

1. `x_q = MXFP4(x)` along `hidden_size`.
2. `gate = x_q W1[e, :2048]^T`, `up = x_q W1[e, 2048:]^T` (exact sums).
3. `h = bf16(silu(gate) * up)`.
4. `partial_k = MXFP4(h) W2[e]^T`.
5. `out = bf16(sum_k w_k * partial_k)`.

`MXFP4(.)` is AITER's dynamic MXFP4 quantizer (`_mxfp4_quant_op`, also the quantizer of AITER's
production `fused_dynamic_mxfp4_quant_moe_sort`): the block maximum is rounded to a power of two (up
when its mantissa is at least 1.75), the scale is that power over 4, and each scaled value is rounded to
the nearest E2M1 value with ties to the even code, saturating at 6. `mxfp4_numerics.py` implements it in
float64. The source task's reference used the "smallest power of two that does not saturate" scale
rule, which no AITER kernel applies to activations; it compared at 13 dB SQNR, which hid that mismatch.
Every product of two E2M1 values with E8M0 scales, and every partial sum at these magnitudes, is exact
in float32, so `gate`, `up` and `partial` of a float32-accumulating kernel equal the reference's.
`reference.py` computes them in float64 per expert, rounds `h` to bfloat16 through float32 as a kernel
does, and accumulates the slots in float64.

## Inputs

Hidden states are standard normal. Weights are drawn in bfloat16 at `1/sqrt(fan_in)` scale and
quantized with the same AITER rule (the source task used its non-saturating rule; this only changes which
valid codes the weights hold); the router draws standard-normal logits over all 65 experts and keeps the top-9 softmax probabilities (not
renormalized), as the source task did. Weights are passed in plain row-major layout (codes
`[expert, out, in / 2]`, two per byte with the even element in the low nibble, scales
`[expert, out, in / 32]`), not the CK/ASM preshuffle of the source task, because the baseline reads
row-major operands; they are uint8 tensors rather than `float4_e2m1fn_x2`. The source task's
`activation` and `doweight_stage1` arguments are fixed by the scenario (SiLU, weights in stage 2), and
`out` is preallocated instead of returned.

## Baseline

AITER's `fused_moe` serves this shape on gfx950 with FlyDSL MXFP4 MoE kernels on AITER's preshuffled
layouts, with its sorting and fused quantize-and-sort kernels (AITER at the pinned commit has no tuned entry for 65
experts and falls back to its heuristic FlyDSL a4w4 kernels); that path is not vendored. The baseline is
AITER's own Triton MXFP4 MoE path (ROCm/aiter `652f2d4b`, MIT, `baseline/LICENSE`):
`dynamic_mxfp4_quant` for the activations and the intermediate, `_fused_moe_kernel_mxfp4_silu` for stage 1,
`_fused_moe_kernel_mxfp4` (with the routing weight) for stage 2 and `reduce_grouped` to add the 9 slots of
a token in float32 (stage 2 stores each slot in bfloat16 first, one rounding more than the reference),
with AITER's gfx950 `MX_FP4` MoE tile table. The kernels are vendored unchanged except
for one line: upstream's stage-1 epilogue rounds `gate` and `up` to bfloat16 before the SiLU; here it
applies `SiLU(gate) * up` to the float32 accumulator, as the production FlyDSL stage 1 and the source
task's reference do. Not taken from AITER: the token-to-expert alignment (`routing.py`), written for this
task with atomic counters because AITER's Triton `moe_align_block_size` walks the tokens in a sequential
loop per expert (production uses a HIP/FlyDSL sorting kernel); it produces the same sorted-ids /
expert-ids / padded-count contract. The baseline may be slower than the production FlyDSL path; it is
the task's baseline.

## Comparison

`out` uses the `baseline_relative` bound with `max_violation_fraction: 0.01` (the default allows none);
`k` keeps its default of 2.

The intermediate requantization is discontinuous. `gate` and `up` are exact, but `silu(gate) * up` is
evaluated in float32, and correct float32 formulations (Triton's `x / (1 + exp2(-x log2(e)))`,
`x * sigmoid(x)`, `x / (1 + exp(-x))`, a reciprocal times `x`) round `h` to a different bfloat16 value
for a few elements per million. When such an element sits on an E2M1 rounding tie, one code of
`MXFP4(h)` moves by one grid step (or, for a block maximum, the block scale changes), and the whole token
row of `out` moves by a few tenths of a percent of its RMS. A CPU simulation of these formulations over
614,400 routes of 2048 standard-normal `gate`/`up` pairs found 2e-6 to 5e-6 of the bfloat16 `h` values and
0.7e-8 to 3.3e-8 of the codes differing from the baseline formulation's, so a token row (9 routes)
differs from the baseline's codes with probability of up to about 6e-4. In a CPU emulation of this layer
with one such flip in every row of 256 tokens, the median row had 0.9% of its elements beyond the k = 2
bound, but 10% of the rows more than 30% (block-scale changes and large routing weights). The expected
handful of flipped rows in a 4096-token case is far below 1% of the output, while without an allowance a
correct candidate with another SiLU formulation would fail almost surely. The allowance still rejects a
badly flipped row in the smallest cases, where one row is 1/64 to all of the output; summed over the cases
and both draws, a candidate whose SiLU rounds differently from the baseline's has an estimated 1%
chance of failing a case (the draws are fixed per task and case, so a given formulation either always
or never hits one). A larger `k` would lower that (to about a third at k = 4) but would also admit
less accurate implementations, so it stays at 2.

The CPU emulation (256 tokens, this bound) shows the baseline at 51.5 dB SQNR against the reference; a
float32 sum of the slots and another SiLU formulation pass with no violations. Wrong variants fail with
most elements out of bound: `h` quantized without its bfloat16 rounding (28.6 dB, 79% of elements),
`gate` and `up` rounded to bfloat16 before the SiLU as upstream's Triton epilogue does (29.3 dB, 76%), no
intermediate quantization (16.9 dB, 95%), the source task's non-saturating scale rule for the activations
(17.1 dB, 95%), the output shifted by 16 bfloat16 ulps (93%) and the output rounded to fp8 (68%). A single
dropped (token, slot) contribution changes one row, 0.4% of a 256-token output, and is accepted there; a
systematic mistake of that kind also changes a third to all of the 1-, 3- and 4-token outputs and fails
them.
