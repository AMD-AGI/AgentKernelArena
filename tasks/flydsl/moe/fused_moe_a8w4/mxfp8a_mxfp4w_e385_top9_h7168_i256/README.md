# Kimi-K2 A8W4 MoE (MXFP8 activations x MXFP4 weights), 385 experts, top-9, hidden 7168, intermediate 256

The routed-expert layer of Kimi-K2 on one tensor-parallel-8 rank with FP4 expert weights and FP8 activations
(AITER's "fp8fp4" / A8W4 MoE), with the shared expert fused into the routed experts as expert 384 (the ninth
slot of every token). The router decisions (`topk_ids`, `topk_weights`) are inputs; the operator is everything
after the router:

```
a1     = mxfp8(x)                                      # per token, 1 x 32 blocks
h      = dequant(a1[t]) @ dequant(w1[e])^T             # 512 outputs: gate = h[:256], up = h[256:]
act    = silu(gate) * up
a2     = mxfp8(act)                                    # per (token, slot), 1 x 32 blocks
out[t] = sum_k topk_weights[t, k] * (dequant(a2) @ dequant(w2[e])^T),   e = topk_ids[t, k]   (bfloat16)
```

`mxfp8` is the operator's own activation quantization and is part of the timed work. Per 32-element block of
float32 values: take the biased float32 exponent of `max|x|`, rounded up when its mantissa is at least 1.5
(bit 22 set), subtract 8 and clamp at 0 to get the e8m0 byte; the codes are `float8_e4m3fn(x * 2^(127 - e8m0))`
rounded to nearest even (at most 384 in magnitude) and the block value is `code * 2^(e8m0 - 127)`. The weights
are MXFP4: e2m1 codes packed two per byte along the contracted axis (low nibble = even element) and one e8m0
byte per 32 contracted elements. The kernel-facing weights and scales use the preshuffled layouts the AITER
kernels read (see below); the reference reads the logical row-major layout.

## Production origin and scenario

Kimi-K2 (7168 hidden, 384 routed experts with top-8, one shared expert, MoE intermediate 2048, i.e. 256 per
rank at TP8) served with fp8 activations and fp4x2 weights: `aiter.fused_moe` with `QuantType.per_1x32`, tuned
in `aiter/configs/model_configs/kimik2_fp8fp4_{tuned,untuned}_fmoe.csv` (256 compute units, 7168/256 with
384/top-8 and with the fused shared expert, 385/top-9), which AITER runs with its FlyDSL A8W4 two-stage
kernels. The source task (`torch2flydsl/moe_a8w4_kernel`) held this FlyDSL implementation with a Kimi-K2
16-token case (384 experts, top-8) and a 32-token 385-expert case of another model; this task fixes the Kimi-K2
geometry with the fused shared expert (385 experts, top-9, intermediate 256).

## Inputs and layout

`inputs.py` quantizes normal weights (standard deviation 0.02 for the 7168-wide gate/up projection, 1/16 for
the 256-wide down projection) to MXFP4 with the source task's rule (scale byte = exponent of `max|w| / 4`
rounded up at 1.5x, e2m1 codes rounded to nearest even); the routing is Kimi-K2's: sigmoid scores of 384
experts in a single group, the 8 best bias-corrected scores, weights = the selected sigmoid scores renormalized,
and the fused shared expert 384 with weight `1 / routed_scaling_factor` (1 / 2.827), as SGLang's fused shared
experts produce it. Hidden states are unit normal bfloat16, except in the range-control case below.

`to_kernel_layout` is the load-time weight preparation of the source task's AITER path, reproduced from
ROCm/aiter at commit 652f2d4b (verified byte for byte against its `aiter/ops/shuffle.py`):
`shuffle_weight(w1, (16, 16))`, `e8m0_shuffle(w1_scale)`, `shuffle_weight_a16w4(w2, 16, False)` and
`shuffle_scale_a16w4(w2_scale, E, False)`. Shapes are unchanged.

## Workloads

Token counts from AITER's Kimi-K2 fp8fp4 tune file: decode batches of 4, 16, 64 and 256 and prefill chunks of
1024, 2048, 4096 and 16384 tokens are timed. One token (nine experts, about 26 MB of weights) is expected below
the 10 us floor and is correctness-only, as are 3, 17, 37, 200, 1000 and 3000 tokens; together they exercise
the three paths of the sorting kernel (up to 16 tokens, up to 2048, beyond). `t2_range` keeps the source
task's bfloat16 output-range control: its hidden states have standard deviation 512, so the bfloat16 stage-1
activation and about 80 % of the output elements exceed the float16 range (65504); an implementation that
routes an intermediate or the output through float16 overflows there. The largest case stays at 16384
tokens: the per-slot stage-2 buffer (tokens x 9 x 7168 bfloat16, 2.1 GB) must stay addressable by the kernels'
32-bit buffer offsets.

## Reference

`reference.py` quantizes both activations with the rule above and evaluates the rest in float64 on the
dequantized operands; `silu(gate) * up` is not rounded to bfloat16 before its quantization.

## Baseline

The baseline is the source task's FlyDSL implementation, with every step a FlyDSL kernel of this package:

1. `moe_sorting.py`: FlyDSL's MoE sorting kernel (blocks of 32 rows, packed `(slot << 24) | token` ids and the
   sorted routing weights);
2. `mx_ops.py` `mxfp8_quant_rows` and `mx_scale_sort`: MXFP8 hidden states and their e8m0 scales gathered into
   AITER's sorted tiled scale layout (`moe_mxfp4_sort`);
3. `moe_gemm.py` stage 1 (`compile_mixed_moe_gemm1`, 32 x 256 x 256 tiles, separated gate/up): the fp8 x fp4
   gate/up GEMM with `silu(gate) * up` in the epilogue, written in bfloat16;
4. the same two `mx_ops.py` kernels requantize the stage-1 result to MXFP8;
5. stage 2 (`compile_mixed_moe_gemm2`, per-slot "reduce" output): the fp8 x fp4 down GEMM times the routing
   weight, one bfloat16 row per (token, slot); as in the source launcher, the grid is persistent (one CTA per
   compute unit) above 256 M blocks;
6. `mx_ops.py` `topk_sum`: the nine rows of every token summed in float32.

The source task also quantized and shuffled the bfloat16 weights inside every call and used AITER host
operators (a HIP MXFP8 quantizer, the CK sorting kernel, a Triton scale sort) and a PyTorch top-k sum. Weight
preparation is load-time work in production and is now the layout transform; the host operators are replaced
by the FlyDSL kernels above (the quantization and sorted-scale kernels follow FlyDSL's `silu_and_mul_fq`), so
that the baseline meets the FlyDSL candidate rules. The GEMM kernels, their configuration and the source
launcher's choice of persistence are unchanged.
The baseline rounds the stage-1 activation and the per-slot outputs to bfloat16.

## Comparison bound

`out` uses `baseline_relative` with `k: 4` and `max_violation_fraction: 0.01` instead of the default (k 2, no
violations). The stage-1 activation is requantized to e4m3 (three mantissa bits), so the rounding an
implementation applies before that quantization decides which codes cross a rounding midpoint: the baseline's
error (about 1.7 % relative RMS, 35 dB SQNR) is made of such code flips, and a legitimate implementation with a
different rounding sequence produces independent flips of the same size, which fail the default bound on 2-4 %
of the elements. A CPU emulation at the real hidden and intermediate sizes (25 or 49 experts, 96 or 128
tokens, four seeds) measured these violation fractions:

| Implementation variant | k = 2 | k = 4 |
| --- | --- | --- |
| float32 activation quantized in the stage-1 epilogue (AITER's fused fp8 output) | 0 | 0 |
| different float32 accumulation order | 0 | 0 |
| float32 partials summed without per-slot rounding | 0 | 0 |
| bfloat16 atomic top-k accumulation, alone or with the fused fp8 output | 1.9e-5 - 2.1e-4 | 0 - 2.2e-6 |
| split-K style: gate/up rounded to bfloat16, activation in float32 | 1.9e-2 - 3.0e-2 | 1.2e-3 - 3.5e-3 |
| gate/up and activation both rounded to bfloat16 | 2.3e-2 - 3.9e-2 | 1.5e-3 - 4.5e-3 |
| error: activation not requantized | 0.16 - 0.19 | 2.6e-2 - 3.4e-2 |
| error: output rounded to fp8 | 0.16 - 0.18 | 3.6e-2 - 4.3e-2 |
| error: hidden states not quantized | 0.44 - 0.47 | 0.18 - 0.20 |
| error: shared-expert weight 1 | 0.97 | 0.94 - 0.95 |

The bound accepts every legitimate variant (the worst at 4.5e-3, 2.2x below the limit) and rejects every
wrong-semantics control by at least 2.6x. The same emulation with the range-control inputs (hidden standard
deviation 512) gives at most 8.6e-4 for the legitimate variants and at least 2.8e-2 for the controls. It does
not reject an output gain error below the operator's own
quantization noise (a 1 % gain error violates 2.5e-5 - 7.7e-5 at k = 4). Because e4m3 is a floating-point
format, any power-of-two block scale that keeps the codes out of the e4m3 subnormal range dequantizes to the
same values; a different e8m0 rounding rule (for example rounding `max|x| / 448` up) only changes elements in
that range and is measured as equivalent.
