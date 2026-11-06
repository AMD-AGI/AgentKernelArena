# Fused MXFP4 MoE (a4w4), 257 experts, top 9, hidden 6144, intermediate 512

The routed-expert layer of AITER's `fused_moe` with `QuantType.per_1x32` MXFP4 weights and MXFP4
activations quantized inside the operator, SiLU gating and the routing weights applied after the down
projection. For each token `t` and slot `k` with expert `e = topk_ids[t, k]`:

1. `x = mxfp4(hidden_states[t])` in 1x32 blocks of the hidden axis;
2. `gate = x . w1[e, :512]`, `up = x . w1[e, 512:]` with dequantized weights (these sums of e2m1 x e2m1 x
   e8m0 products are exact in float32 at these magnitudes, whatever the accumulation order);
3. `h = bfloat16(silu(gate) * up)` with SiLU in float32, then `h_q = mxfp4(h)` in 1x32 blocks of the
   intermediate axis;
4. `y = topk_weights[t, k] * (h_q . w2[e])`; `out[t] = bfloat16(sum_k y)`.

`mxfp4()` is AITER's dynamic MXFP4 quantization with its default scale rule (`MxScaleRoundMode::RoundUp`,
`csrc/include/mx_quant_utils.h`): the block scale is `2^s`, the smallest power of two with
`amax <= 6 * 2^s` (`ceil_pow2(amax / 6)`), and every value divided by the scale rounds to the nearest e2m1
value (0, 0.5, 1, 1.5, 2, 3, 4, 6), ties to the even code. Weights are e2m1 codes, two per byte (even index in
the low nibble), with one e8m0 scale byte (`2^(s - 127)`) per 32 values of the reduction axis, in the plain
row-major layout: `w1 [257, 1024, 3072]` (gate rows, then up rows) with `w1_scale [257, 1024, 192]`,
`w2 [257, 6144, 256]` with `w2_scale [257, 6144, 16]`.

**Scenario and production origin.** The GLM-5 MoE layer (hidden 6144, 256 routed experts plus the shared
expert fused as expert 256, top 9) at 4-way tensor parallelism, expert intermediate 512. AITER ships tuned
MXFP4 configurations for exactly this layer (`aiter/configs/model_configs/glm5_fp4_tuned_fmoe.csv`, model_dim
6144, inter_dim 512, 257 experts, top 9, 1 to 32768 tokens). The old task (`mxfp4_moe_e257_i256`) named it after
the packed byte width of a `w2` row (256); the logical intermediate size is 512 (`w1` rows 1024 = 2 x 512,
`w2_scale` 16 blocks of 32), so this task is named `i512`.

**Numerics.** The semantics are the old task's reference, which are those of AITER's production a4w4 path
for a bfloat16 stage-1 output: the activations and the bfloat16 intermediate are quantized by AITER's HIP
`fused_dynamic_mx_quant_moe_sort` with the default RoundUp scale (`MX_DEFAULT_ROUND_MODE`, also the FlyDSL
quantizer's default), and AITER's FlyDSL stage 1 applies SiLU to its float32 gate/up accumulators. Most of
AITER's tuned entries for this exact shape do not take that path: at 1 and at 8 to 8192 tokens they select
FlyDSL stage-1 kernels with a fused MXFP4 epilogue (`flydsl_moe1_afp4_wfp4_bf16_*_fp4`), which quantize the
float32 intermediate without rounding it to bfloat16 and with a different block-scale rule (amax rounded to
the nearest power of two). This task fixes one semantics, the old task's reference; an implementation of that
fused epilogue combines the float32-intermediate and scale-rule deviations measured below and does not pass.

**Reference.** `reference.py` quantizes the activations and the intermediate with an independent float64
implementation of the rule above, dequantizes the weights from codes and scales and computes every product and
sum in float64 per expert (chunked to bound memory); `h` is rounded to bfloat16 through float32, as a float32
kernel does.

**Inputs.** `inputs.py` draws normal expert weights with std `1/sqrt(fan-in)`, rounds them to bfloat16 and
quantizes them with the same RoundUp rule, as the old task's initializer did; hidden states are standard normal;
the router is the old task's: normal logits over the 257 experts, the top 9 of their softmax (distinct ids,
probabilities not renormalized). It does not pin the fused shared expert into every token's last slot as
production routing does.

**Workloads.** Token counts from AITER's tuned list: 1, 4, 16, 64, 256, 1024, 4096 and 8192 tokens are timed
(8192x); 3, 77 and 1000 tokens are correctness-only boundary cases. The weights are 1.29 GB (5.0 MB per
expert), so a single token (9 experts, 45 MB) is estimated at about 14 us by the 4.5 TB/s + 4 us rule; from
about 256 tokens every expert is read. The layer does 170 MFLOP per token; at 8192 tokens the float64
reference does 1.4 TFLOP and the baseline's per-slot buffer is 0.9 GB, so larger prefill chunks are left out to
keep evaluation time reasonable.

**Baseline.** AITER's `fused_moe` serves this shape with FlyDSL MoE kernels on AITER's preshuffled layouts
(the tuned entries name `flydsl_moe1_afp4_wfp4_bf16_*` and `flydsl_moe2_layout_afp4_wfp4_bf16_*`) and a HIP
quantize-and-sort kernel; that path (tens of thousands of lines of FlyDSL kernel builders, sorting and
dispatch, plus HIP) cannot be vendored as one language. The baseline is AITER's own Triton MXFP4 MoE:
`_fused_moe_kernel_mxfp4_silu` (stage 1), `_fused_moe_kernel_mxfp4` (stage 2, routing weight applied),
`_mxfp4_quant_kernel` for both quantizations and the gfx950 `MOE-MX_FP4` tile table, in the two-stage sequence
these kernels are written for (AITER tests each stage on its own in
`op_tests/triton_tests/moe/test_moe_mx.py`). Changes, listed in `task.yaml`, align
it with the production numerics above: the quantizer's block scale uses AITER's RoundUp rule instead of the Triton
kernel's EVEN rule (`amax` rounded to a power of two at a mantissa of 1.75, divided by 4), and stage 1 computes
SiLU(gate) * up from its float32 accumulator (upstream rounds gate and up to bfloat16 first). The unit per-tensor
scales are removed; token-to-expert alignment and the final top-k sum are Triton kernels written for this task
(`baseline/routing.py`), since vLLM's alignment is a CUDA op and AITER's Triton alignment loops sequentially over
every token per expert. Stage 2 writes one bfloat16 row per (token, slot) and the sum adds them in float32, so the
baseline rounds each slot once more than the reference. `baseline/LICENSE` is the AITER MIT license. The old
task timed the installed `aiter.fused_moe`; speedups here are relative to the Triton kernels, which are expected
to be slower than AITER's FlyDSL path.

**Comparison: `sqnr` with `min_db: 40` instead of the default bound.** The intermediate requantization is
discontinuous. When an implementation's float32 SiLU differs from another's in the last bits (any two exp2/
division formulations do), `h` occasionally rounds to the neighbouring bfloat16 value, and when that pair
straddles an e2m1 decision boundary one code of `h_q` changes, which moves the whole output row of that token by
up to a few percent. A CPU emulation of four SiLU formulations over 2e8 intermediate elements found such changes
at 2.5e-8 to 5.5e-8 per element, about 2.5e-4 per token, so a correct candidate is expected to differ from the
reference (and from the baseline, at other positions) in several codes over the workload's two draws. The default
`baseline_relative` bound fails a case on a single such change, and a violation fraction does not rescue small
batches: one change violates 18% of its token row on average (up to 83%). The output SQNR is robust to it: a
token row with one changed code has a median SQNR of 46 dB (17% of the changes fall below 40 dB, which only
matters for batches of a few tokens), giving a correct candidate a failure probability of about 0.1% over the
workload. The threshold still rejects every
deviating semantics by about 10 dB: the emulated baseline scores 51.6 dB and a float32 top-k sum about 99 dB
(against the bfloat16-rounded reference), whereas rounding gate and up to bfloat16 before SiLU (the upstream Triton
epilogue) scores 29.8 dB, quantizing the float32 intermediate without the bfloat16 rounding 27.7 dB, the EVEN scale
rule 21.6 dB, and the auditor's fp8-rounded and 16-ulp-shifted controls 31.2 dB and 24.1 dB. Non-finite or
unwritten elements still fail. The old task's SQNR >= 13 dB gate is replaced.

**Interface changes from the old task.** Weights and scales are uint8 in the plain row-major layout read by the
Triton kernels (the old task passed `float4_e2m1fn_x2` weights in AITER's preshuffled tile layout to
`aiter.fused_moe`).
