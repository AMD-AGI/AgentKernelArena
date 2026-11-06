# MXFP4 routed MoE (a4w4), GLM-5 per TP8 rank: 257 experts, top-9, hidden 6144

The routed-expert layer of GLM-5 served with MXFP4 weights and MXFP4 activations, one tensor-parallel rank
of eight: 256 routed experts plus the shared expert fused in as expert 256, top-8 routing plus the shared
slot (top-9), hidden size 6144, expert intermediate size 2048 / 8 = 256, SiLU gating, bfloat16 in and out.
This is AITER's `fused_moe` with `QuantType.per_1x32` and fp4x2 activations and weights; the shape and the
token counts are those of AITER's tuned table `aiter/configs/model_configs/glm5_fp4_tuned_fmoe.csv`. The
variable axis is the batch's token count.

The directory name follows the source task's definition
`fused_moe_per_1x32_d6144_e257_topk9_n512_k3072_i128`, whose `i128` is the packed width of `w2` (two e2m1
values per byte); the logical intermediate size is 256.

## Operator

Per token `t` and slot `j` (expert `e = topk_ids[t, j]`):

```
g      = mxfp4(hidden_states[t]) @ W1[e]^T          # 512 outputs: gate = g[:256], up = g[256:]
act    = bfloat16(silu(gate) * up)
out[t] = bfloat16(sum_j topk_weights[t, j] * (mxfp4(act) @ W2[e]^T))
```

`mxfp4` is the operator's activation quantizer, AITER's production MXFP4 quantization in its default
RoundUp scale mode (`kDefaultMxScaleRoundMode` in `csrc/include/mx_quant_utils.h`, used by the HIP
`fused_dynamic_mx_quant_moe_sort` that `fused_moe` calls for both stages and by the FlyDSL MoE kernels'
fused quantization): per block of 32 values, the e8m0 scale is `ceil_pow2(max(amax, 1e-10) / 6)`, so no
value saturates, and each value divided by the scale rounds to the nearest e2m1 value with ties to even (the
gfx950 `cvt_scalef32_pk_fp4` conversion). Both quantizations, including the bfloat16 rounding of the
SiLU-and-mul output before it is requantized, are part of the operator and are reproduced exactly by
`reference.py`; everything else runs in float64. This is also the quantizer of the source task's reference;
a CPU cross-check of `reference.py` against that reference (on the source task's weight layout, with a
17-expert subset) agrees to the bfloat16 output rounding.

Weights are MXFP4: `w1 [E, 512, 3072]` and `w2 [E, 6144, 128]` hold e2m1 codes packed along the contracted
axis (low nibble first), with one e8m0 scale byte per 32 contracted values. `inputs.py` draws them as the
source task did: `N(0, 1/fan_in)` in bfloat16, quantized with upward-rounded scales and round-to-nearest-even
codes. The kernel-facing layout (`to_kernel_layout`, weight preparation done once at load time) interleaves
the gate and up rows of `w1` and stores both scale tensors in the gfx950 MX-scale tile order of AITER's
`shuffle_scale_moe`. Routing follows SGLang's fused-shared-expert convention for GLM-5: the top-8 of 256
sigmoid router scores, renormalized to sum to one, and the shared expert 256 in the ninth slot with weight
`1 / routed_scaling_factor = 0.4`, so expert 256 receives every token. (The source task drew a softmax top-9
over all 257 experts without renormalization, which never models the shared expert.)

## Baseline

AITER dispatches this shape on gfx950 to FlyDSL two-stage MoE kernels
(`flydsl_moe1_afp4_wfp4_bf16_*`, `flydsl_moe2_afp4_wfp4_bf16_*`) chosen by kernel name from the tuned table,
with a FlyDSL sorting kernel and HIP quantization. Those FlyDSL sources (more than 8000 lines with their
helpers) are tied to AITER's dispatch and chip queries and pinned to flydsl 0.3.0, so they were not
vendored. The baseline is AITER's Triton MXFP4 x MXFP4 MoE path instead (all Triton, vendored at
ROCm/aiter 652f2d4b): routing from the given top-k ids (vLLM's `pack_bitmatrix` plus AITER's routing
kernels), `mxfp4_quant` of the tokens, the a4w4 GEMM with gather and the fused SiLU-and-mul epilogue (the
triton_kernels SwiGLU with alpha 1, no clamp and no residual), `mxfp4_quant` of the bfloat16 intermediate,
the gate-weighted a4w4 down GEMM and the grouped top-k reduction.

One kernel change aligns the baseline with the production operator: the upstream Triton `_mxfp4_quant_op`
rounds the block maximum to the nearest power of two and divides it by 4 (saturating values when the maximum
lies in `(1.5, 1.75) * 2^k`, roughly a fifth of the blocks), whereas AITER's production quantizers use the
RoundUp scale above. `_mxfp4_quant_op` computes the RoundUp scale (`amax * (1/6)` in float32, rounded up to
a power of two, as the HIP kernel does); its nearest-even e2m1 conversion is unchanged. A bit-level CPU
emulation of the modified op matched `reference.py`'s quantizer on 1e8 bfloat16 values, half of them drawn
on quarter steps of a power of two so that many fall exactly on rounding midpoints. All other modifications
are listed in `task.yaml`. The baseline is expected to be slower than the FlyDSL production kernels; it is
the task's baseline.

## Workloads

Token counts 1, 4, 16, 64, 256, 1024, 4096 and 16384 of the tuned table are timed (the smallest, nine
experts, still moves about 22 MB of weights through about ten kernel launches). Odd and unaligned batches
(2, 3, 17, 100, 1000, 8000) are correctness-only; 17 crosses the fused-routing bound (16 tokens) and 8000
exceeds the 65535 routes that 16-bit routing indices hold.

## Comparison

`out` uses the baseline-relative bound with `max_violation_fraction: 0.01`. The float32 GEMM accumulation is
exact here (the products of e2m1 values and power-of-two scales, and their sums, fit float32), but the SiLU
is evaluated in float32 and its bfloat16 rounding feeds the second MXFP4 quantization. A CPU measurement
over 2,000,000 routes found that three equivalent float32 SiLU formulas (the baseline's `exp2` form, `x *
sigmoid(x)` and `x / (1 + exp(-x))`) each round about 7e-6 of the bfloat16 intermediates differently from
the float64 value, and change an MXFP4 code in about 1e-5 of the routes. Such an isolated code change moves
one token's output row by a few tenths of a percent of its magnitude (about 1 % for the shared slot, whose
weight is 0.4), far above that row's rounding-level baseline error. Without an allowance a correct candidate
whose SiLU rounds differently from the baseline's would fail the larger cases (16384 tokens hold 147456
routes); the 1 % allowance absorbs such isolated rows from about 64 tokens on. Systematic errors still fail
at every size: in the CPU emulation, skipping the bfloat16 rounding of the intermediate, the upstream Triton
scale rule and round-half-up ties each put 36 % to 95 % of the elements outside the bound (output SQNR 16 to
30 dB against about 53 dB for the baseline); the QA controls' 16-ulp shift (6 %) and float8 rounding change
every element by more than the bound. The residual risk is in the cases of 17 tokens or fewer: there one
code change exceeds the 1 % allowance, which a candidate with a different SiLU rounding meets with a
probability of about one percent over those cases' two input draws.