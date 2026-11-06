# Fused MXFP4 MoE (a4w4), 33 experts, top 9, hidden 6144, intermediate 2048

The routed-expert layer of AITER's `fused_moe` with `QuantType.per_1x32` MXFP4 weights and MXFP4
activations quantized inside the operator, SiLU gating and the routing weights applied after the down
projection. For each token `t` and slot `k` with expert `e = topk_ids[t, k]`:

1. `x = mxfp4(hidden_states[t])` in 1x32 blocks of the hidden axis;
2. `gate = x . w1[e, :2048]`, `up = x . w1[e, 2048:]` with dequantized weights (these sums of e2m1 x e2m1 x
   e8m0 products are exact in float32 at these magnitudes, whatever the accumulation order);
3. `h = bfloat16(silu(gate) * up)` with SiLU in float32, then `h_q = mxfp4(h)` in 1x32 blocks of the
   intermediate axis;
4. `y = topk_weights[t, k] * (h_q . w2[e])`; `out[t] = bfloat16(sum_k y)`.

`mxfp4()` is AITER's dynamic MXFP4 quantization with its default scale rule (`MxScaleRoundMode::RoundUp`,
`csrc/include/mx_quant_utils.h`): the block scale is `2^s`, the smallest power of two with
`amax <= 6 * 2^s` (`ceil_pow2(amax / 6)`), and every value divided by the scale rounds to the nearest e2m1
value (0, 0.5, 1, 1.5, 2, 3, 4, 6), ties to the even code. Weights are e2m1 codes, two per byte (even index in
the low nibble), with one e8m0 scale byte (`2^(s - 127)`) per 32 values of the reduction axis, in the plain
row-major layout: `w1 [33, 4096, 3072]` (gate rows, then up rows) with `w1_scale [33, 4096, 192]`,
`w2 [33, 6144, 1024]` with `w2_scale [33, 6144, 64]`.

**Scenario and production origin.** The old task's `mxfp4_moe_e33_i1024` scenario: one rank of a
256-routed-expert MoE layer with hidden 6144 and expert intermediate 2048 (the GLM-5 layer that AITER tunes as
6144/2048/257/9 in `aiter/configs/model_configs/glm5_fp4_tuned_fmoe.csv`) under 8-way expert parallelism, 32 local
routed experts plus the fused shared expert, nine slots per token. AITER tunes the same per-rank structure for
DeepSeek-V3 at hidden 7168 (`a8w8_blockscale_tuned_fmoe_ds_v3.csv`, 7168/2048/33/9); it ships no tuned MXFP4
entry for 6144/2048/33/9, so its `fused_moe` takes the heuristic FlyDSL path. The old task named the scenario after
the packed byte width of a `w2` row (1024); the logical intermediate size is 2048 (`w1` rows 4096 = 2 x 2048,
`w2_scale` 64 blocks of 32), so this task is named `i2048`.

**Numerics.** The semantics are the old task's reference, which are those of AITER's production a4w4 path
with a bfloat16 stage-1 output (the heuristic path this shape takes): the activations and the bfloat16
intermediate are quantized by AITER's HIP `fused_dynamic_mx_quant_moe_sort` with the default RoundUp scale
(`MX_DEFAULT_ROUND_MODE`, also the FlyDSL quantizer's default), and AITER's FlyDSL stage 1 applies SiLU to its
float32 gate/up accumulators.

**Reference.** `reference.py` quantizes the activations and the intermediate with an independent float64
implementation of the rule above, dequantizes the weights from codes and scales and computes every product and
sum in float64 per expert (chunked to bound memory); `h` is rounded to bfloat16 through float32, as a float32
kernel does.

**Inputs.** `inputs.py` draws normal expert weights with std `1/sqrt(fan-in)`, rounds them to bfloat16 and
quantizes them with the same RoundUp rule, as the old task's initializer did; hidden states are standard normal;
the router is the old task's: normal logits over the 33 experts, the top 9 of their softmax (distinct ids,
probabilities not renormalized), so every slot of a token is local to the rank and the shared expert is not
pinned to a slot.

**Workloads.** The old task's powers of two: 1, 4, 16, 64, 256, 1024 and 4096 tokens are timed (4096x); 3, 77
and 1000 tokens are correctness-only boundary cases. The weights are 662 MB (20 MB per expert), so a single
token (9 experts, 180 MB) is estimated at about 44 us by the 4.5 TB/s + 4 us rule, and from about 16 tokens every
expert is read. The layer does 680 MFLOP per token; at 4096 tokens the float64 reference does 2.8 TFLOP, so the
old task's largest batch is kept as the largest case.

**Baseline.** AITER's `fused_moe` serves this shape with FlyDSL MoE kernels on AITER's preshuffled layouts
(heuristic `flydsl_moe1_afp4_wfp4_bf16_t*x128x256*` stage kernels) and a HIP quantize-and-sort kernel; that path
(tens of thousands of lines of FlyDSL kernel builders, sorting and dispatch, plus HIP) cannot be vendored as one
language. The baseline is AITER's own Triton MXFP4 MoE: `_fused_moe_kernel_mxfp4_silu` (stage 1),
`_fused_moe_kernel_mxfp4` (stage 2, routing weight applied), `_mxfp4_quant_kernel` for both quantizations and the
gfx950 `MOE-MX_FP4` tile table, in the two-stage sequence these kernels are written for
(AITER tests each stage on its own in `op_tests/triton_tests/moe/test_moe_mx.py`). Changes, listed in `task.yaml`, align it with the production
numerics above: the quantizer's block scale uses AITER's RoundUp rule instead of the Triton kernel's EVEN rule
(`amax` rounded to a power of two at a mantissa of 1.75, divided by 4), and stage 1 computes SiLU(gate) * up from
its float32 accumulator (upstream rounds gate and up to bfloat16 first). The unit per-tensor scales are removed;
token-to-expert alignment and the final top-k sum are Triton kernels written for this task (`baseline/routing.py`),
since vLLM's alignment is a CUDA op and AITER's Triton alignment loops sequentially over every token per expert.
Stage 2 writes one bfloat16 row per (token, slot) and the sum adds them in float32, so the baseline rounds each
slot once more than the reference. `baseline/LICENSE` is the AITER MIT license. The baseline package is identical
to that of this operator's 257-expert, intermediate-512 scenario. The old task timed the installed `aiter.fused_moe`; speedups here are
relative to the Triton kernels, which are expected to be slower than AITER's FlyDSL path.

**Comparison: `sqnr` with `min_db: 40` instead of the default bound.** The intermediate requantization is
discontinuous. When an implementation's float32 SiLU differs from another's in the last bits (any two exp2/
division formulations do), `h` occasionally rounds to the neighbouring bfloat16 value, and when that pair
straddles an e2m1 decision boundary one code of `h_q` changes, which moves the whole output row of that token by
up to a few percent. A CPU emulation of four SiLU formulations over 2e8 intermediate elements found such changes
at 2.5e-8 to 5.5e-8 per element, about 1e-3 per token at intermediate 2048, so a correct candidate is expected to
differ from the reference (and from the baseline, at other positions) in several codes over the workload's two
draws. The default `baseline_relative` bound fails a case on a single such change, and a violation fraction does
not rescue small batches, where one change violates a large part of its token row. The output SQNR is robust to
it: a token row with one changed code has a median SQNR of 53 dB (4% of the changes fall below 40 dB, which only
matters for batches of a few tokens), giving a correct candidate a failure probability well below 0.1% over the
workload. The threshold still rejects every deviating semantics by about 10 dB: the emulated baseline scores
51.5 dB and a float32 top-k sum about 96 dB (against the bfloat16-rounded reference), whereas rounding gate and up
to bfloat16 before SiLU (the upstream Triton epilogue) scores 30.1 dB, quantizing the float32 intermediate without
the bfloat16 rounding 28.6 dB, the EVEN scale rule 21.3 dB, and the auditor's fp8-rounded and 16-ulp-shifted
controls 31.5 dB and 24.1 dB. Non-finite or unwritten elements still fail. The old task's SQNR >= 13 dB gate is
replaced.

**Interface changes from the old task.** Weights and scales are uint8 in the plain row-major layout read by the
Triton kernels (the old task passed `float4_e2m1fn_x2` weights in AITER's preshuffled tile layout to
`aiter.fused_moe`).
