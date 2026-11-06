# DeepSeek-V3 FP8 block-scaled MoE, 257 experts, top-9, hidden 7168, intermediate 256

The routed-expert layer of DeepSeek-V3 (and R1) with its FP8 checkpoint on one tensor-parallel-8 rank,
with the shared expert fused into the routed experts (expert 256, the ninth slot of every token). The
router decisions (`topk_ids`, `topk_weights`) are inputs; the operator is everything after the router:

```
a1        = fp8_group_quant(x)                                    # per token, 1 x 128 groups
h         = dequant(a1[t]) @ dequant(w1[e])^T                     # w1: 128 x 128 block scales
act       = silu(h[:256]) * h[256:]
a2        = fp8_group_quant(act)                                  # per (token, slot), 1 x 128 groups
out[t]    = sum_k topk_weights[t, k] * (dequant(a2) @ dequant(w2[e])^T),  e = topk_ids[t, k]
```

`fp8_group_quant` is the operator's own activation quantization and is part of the timed work:
`scale = max(max|x|, 1e-10) / 448` in float32 per 128-wide group, `code = float8_e4m3fn(clamp(x * (1 / scale),
-448, 448))` with round-to-nearest-even. The weights are the checkpoint's float8_e4m3fn codes with one float32
scale per 128 x 128 block (`w1` rows 0-255 are the gate projection, rows 256-511 the up projection). The output
is bfloat16.

## Production origin and scenario

This is the FP8 block-scale ("per_1x128") fused MoE that AITER tunes for DeepSeek-V3 on MI355X: the
7168 / 256 / 257 / top-9 rows of `aiter/configs/model_configs/a8w8_blockscale_tuned_fmoe_ds_v3.csv` (256
compute units, `QuantType.per_1x128`, token counts 1 to 32768). The source tasks were the arena's
`fmoe_fp8_blockscale_g1u1` task (DeepSeek-V3 16- and 32-token batches against AITER's assembly kernel) and an
MI355X serving-trace task of AITER's Composable Kernel two-stage MoE that mixed three quantization
configurations; this task is the FP8 block-scale configuration at the DeepSeek-V3 shape. The MXFP4 and
per-tensor FP8 configurations of that trace are separate operators. Two other FP8 block-scale shapes of the
source tasks are sibling scenarios of this operator, not cases of this one, because hidden size and
expert geometry are constants of a scenario: the `fmoe_fp8_blockscale_g1u1` task's 32-token case with
hidden size 6144 (GLM-5, 257 experts, top-9, intermediate 256; AITER's
`a8w8_blockscale_untuned_fmoe_glm5.csv`) and the trace's MiniMax-M2.7 layer (hidden 3072, 256 experts,
top-8, intermediate 384).

## Inputs

`inputs.py` generates the weights the way a DeepSeek-V3 FP8 checkpoint stores them: normal weights (standard
deviation 0.02 for the 7168-wide gate/up projection, 1/16 for the 256-wide down projection) quantized per
128 x 128 block with `scale = max|w| / 448`. The routing is DeepSeek-V3's group-limited sigmoid routing
(8 groups of 32 experts, the 4 best groups by the sum of their two best bias-corrected scores, the 8 best
experts inside them; weights are the selected sigmoid scores renormalized) with SGLang's fused shared expert
appended as expert 256 with weight `1 / routed_scaling_factor` = 0.4. Hidden states are unit normal bfloat16.

## Workloads

Token counts from the AITER tune file: decode batches 1, 4, 16, 64 and 256, and prefill chunks of 1024, 4096
and 16384 tokens are timed (even one token reads nine experts, about 50 MB of weights). 3, 37, 200, 1000 and
3000 tokens are correctness-only boundary and unaligned cases. They cover the three tile configurations of
the baseline (below 256, below 1024 and from 1024 tokens).

## Reference

`reference.py` quantizes both activations with the operator's rule and evaluates everything else in float64
on the dequantized operands, without rounding the gate/up projection or the activation to bfloat16.

## Baseline

The production kernels for this operator on MI355X are AITER's assembly `fmoe_fp8_blockscale_g1u1` and its
Composable Kernel two-stage MoE; neither can be vendored as source. The baseline is AITER's Triton FP8
block-scale MoE path (ROCm/aiter, MIT), which computes the same quantized operator with six kernel stages:
`fused_flatten_fp8_group_quant` (activation quantization), the four `moe_align_block_size` kernels, the
`fused_moe` grouped GEMM (`use_fp8_w8a8`, `block_shape` [128, 128]) for gate/up, the fused
SiLU-and-mul FP8 requantization, `fused_moe` again for the routing-weighted down projection, and
`reduce_grouped` for the top-k sum. Kernels are copied verbatim; the launchers keep only the branches this
operator takes, and the gfx950 FP8_W8A8 tile table and its token-count selection are inlined. The baseline
rounds the gate/up projection and the per-slot expert outputs to bfloat16, as the Triton path does.

## Comparison bound

`out` uses `baseline_relative` with `k: 4` and `max_violation_fraction: 0.005` instead of the default (k 2,
no violations). The intermediate activation is requantized to FP8 with three mantissa bits, so whether an
implementation rounds the gate/up projection or the activation to bfloat16 first decides which codes land on
the other side of a rounding midpoint. The baseline's own error (about 1.4 % relative RMS, 37 dB SQNR) is
made of these code flips, and a different but legitimate rounding sequence produces independent flips of the
same size: under the default bound it fails on 1-3 % of the elements. A CPU emulation at the real hidden and
intermediate sizes (17 or 33 experts, 96 or 128 tokens, four seeds) measured these violation fractions:

| Implementation variant | k = 2 | k = 4 |
| --- | --- | --- |
| float32 gate/up, fused requantization | 0 | 0 |
| different float32 accumulation order | 0 | 0 |
| float32 gate/up, activation rounded to bfloat16 | 1.1e-2 - 2.6e-2 | 2.1e-4 - 2.8e-3 |
| bfloat16 gate/up and activation (vLLM-style) | 2.3e-2 - 3.1e-2 | 9.7e-4 - 1.7e-3 |
| bfloat16 atomic top-k accumulation | 2.5e-4 - 3.3e-4 | 1.1e-6 - 4.4e-6 |
| error: activation not requantized | 0.17 - 0.19 | 2.3e-2 - 2.9e-2 |
| error: output rounded to fp8 | 0.20 - 0.21 | 5.2e-2 - 5.7e-2 |
| error: power-of-two activation scales | 0.35 - 0.36 | 0.11 - 0.12 |
| error: hidden states not quantized | 0.49 - 0.51 | 0.22 - 0.24 |

The chosen bound accepts every legitimate variant (the worst at 2.8e-3, 1.8x below the limit) and rejects each
wrong-semantics control by at least 4.6x. Like any elementwise bound it does not reject an output gain error
smaller than the operator's own quantization noise (a 1 % gain error violates 1.6e-5 - 1.3e-4 at k = 4).
