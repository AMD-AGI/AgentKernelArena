# GPT-OSS MoE, bf16 tokens x MXFP4 experts, 128 experts, top-4, hidden 3072, intermediate 512

The routed-expert block of GPT-OSS-120B on one tensor-parallel rank. Each token is sent to 4 of 128 experts
(`topk_ids`, with router weights `topk_weights`). For token `t` and slot `j` (expert `e`):

```
g      = x[t] @ W1[e]^T                       # 1024 outputs: gate = g[:512], up = g[512:]
act    = min(gate, 7) * sigmoid(1.702 * min(gate, 7)) * (clamp(up, -7, 7) + 1)
out[t] = sum_j topk_weights[t, j] * (act @ W2[e]^T)        # bfloat16
```

`W1` [128, 1024, 3072] and `W2` [128, 3072, 512] are MXFP4: e2m1 codes packed two per byte along the contracted
axis (low nibble first) and one e8m0 scale byte, `2^(byte - 127)`, per 32 contracted elements. The tokens are
bfloat16 and are not quantized (a16w4). The activation is GPT-OSS SwiGLU (`alpha = 1.702`, limit 7, `up + 1`).

## Production origin

The source task benchmarked AITER's CK-Tile two-stage MoE (`aiter.fused_moe`, `QuantType.per_1x32`,
`ActivationType.Swiglu`, bf16 x fp4x2) on the GPT-OSS-120B shape traced in a vLLM 0.24 MI355X session: 64 tokens,
`model_dim = 3072` (2880 padded for MXFP4), `inter_dim = 512` (2880 / TP8 = 360, padded), 128 experts, top-4.
The routing (`topk_ids`, `topk_weights`) is an input, as in `fused_moe`; GPT-OSS routes with a softmax over its
top-4 router logits. The source task's model has expert biases; its benchmark did not pass them, and neither does
this task.

## Inputs and layout

`inputs.py` draws uniform random e2m1 code bytes and e8m0 scale bytes in a few binades (2^-8..2^-5 for `W1`,
2^-10..2^-7 for `W2`) so that unit-normal tokens give gate/up values with a standard deviation of about 3: the
SwiGLU clamp at 7 is active for about 1% of the gate values and the output has unit scale. Routing takes the top-4
of unit-normal logits (distinct experts) with softmax weights.

The logical weights (what `reference.py` reads) keep the AITER layout: `W1` rows 0..511 are the gate projection
and rows 512..1023 the up projection; scales are `[E, N, K/32]`. The kernel-facing layout (`to_kernel_layout`) is
the weight preparation this Triton kernel reads on gfx950, done once at load time in production:

- `w1` and its scales interleave gate and up rows (row `2i` = gate `i`, row `2i+1` = up `i`), which the fused
  SwiGLU epilogue splits as even/odd columns (GPT-OSS checkpoints store these rows interleaved);
- `w1_scale` [128, 3072, 32] and `w2_scale` [128, 512, 96] are the scales permuted into the gfx950 (CDNA4)
  MX-scale tile order of AITER's `shuffle_scale_moe` (32-row stripes of N, 8-wide groups of K/32), the layout the
  kernel reads with `SWIZZLE_MX_SCALE="CDNA4_SCALE"` (AITER's tests of this kernel use it, and vLLM uses the same
  CDNA4 scale layout for its triton_kernels GPT-OSS path on gfx950 at tensor-parallel sizes up to 2). At
  tensor-parallel 4 and above, which includes this TP8 shape, vLLM keeps the scales unswizzled, because its A8W4
  dispatch may pick K tiles below the 256 that the swizzle needs. This a16w4 kernel always uses 256-wide K tiles,
  so the swizzled layout is valid here. The
  `unswizzle_mx_scale_cdna4` step in `baseline/moe_gemm_a16w4.py` is its inverse inside each tile.

`w1` and `w2` keep their `[E, N, K/2]` shape; the baseline views them as column-major `[E, K/2, N]`.

## Reference

`reference.py` dequantizes each routed expert in float64, applies the clamped SwiGLU and the weighted top-k sum in
float64 with no intermediate rounding. The baseline (and AITER's own path) stores the SwiGLU output and the
per-slot weighted GEMM2 rows in bfloat16 before the final sum; that rounding is part of the baseline's error, so a
candidate that keeps those intermediates in higher precision is not penalized. The default `baseline_relative`
bound applies.

## Baseline

The CK-Tile kernel of the source task cannot be vendored as source, so the baseline is AITER's Triton bf16 x MXFP4
MoE GEMM, `aiter/ops/triton/moe/moe_op_gemm_a16w4.py` (`ROCm/aiter` at commit
652f2d4b768239aefd515c929e84817681ff598d, MIT; a port of the triton_kernels `matmul_ogs` kernels vLLM runs
GPT-OSS with), together with AITER's routing kernels. All of it is Triton:

1. `pack_bitmatrix` (vLLM `make_routing_data`, `vllm-project/vllm` at commit
   edbc4969a76b25c531f7b9bb16b79984e65ef023, Apache-2.0) packs `topk_ids` into a token x expert bitmatrix.
2. AITER's `sort_tokens` / `sort_tokens_fused` (the latter for at most 16 tokens, as in AITER `routing()`) compute
   the expert histogram, token and tile offsets, the GEMM tile map and the gather/scatter permutations with the
   router weights in expert order. AITER's `routing()` takes router logits and runs its own top-k kernel first;
   here the top-k is given, so that kernel is replaced by `pack_bitmatrix` and the scratchpads it zeroes are
   allocated with zeros.
3. `_moe_gemm_a16w4` GEMM1 gathers token rows per expert and applies the SwiGLU in its epilogue; GEMM2 multiplies
   by the router weight; `_reduce_grouped` sums each token's 4 rows into `out`. Block sizes follow AITER's
   `get_kernel_config` (block_m from tokens per expert, 16..128) and the routing `block_m` heuristic of AITER.
   vLLM's AITER W4A16 MoE backend (`aiter_triton_kernel_w4a16_moe_forward`,
   `vllm/model_executor/layers/fused_moe/experts/aiter_mxfp4_w4a8_moe.py`) makes the same two GEMM calls after
   routing from router logits.

The kernels are unchanged apart from local imports, the removed profiling-only `launch_metadata` hook and one
parameter renamed from the builtin name `input`. The bias inputs are absent, the gfx1250 tensor-descriptor paths
are disabled, and the reduction writes into the preallocated `out`. `baseline/LICENSE` is the AITER MIT license;
`baseline/pack_bitmatrix.py` keeps its vLLM Apache-2.0 header.

## Workloads

Token counts from decode batches to prefill chunks: 8 to 16384 tokens are timed (2048x). The weights dominate
small batches: 8 tokens already touch about 29 of the 128 experts (about 70 MB of weights, an estimated 19 us).
Batches of 1-3 tokens touch at most 12 experts and would sit near the 10 us floor, so they are correctness-only,
together with unaligned batches (17, 33, 100, 1000, 3000). 17 tokens is the first batch past the fused-routing
bound (at most 16 tokens); 16384 tokens is the first with more than 65535 routed rows (32-bit permutation
indices).

The routing kernels take their shapes from `topk_ids` only and nothing on the host reads GPU data, so the six
launches (plus the zero fills of the routing scratchpads) are capturable in a CUDA graph. The kernels use
`tl.dot_scaled` (bf16 x e2m1 with `fast_math`), `tl.reduce_or`, `tl.sort` and `tl.associative_scan` as AITER does
at the recorded commit.
