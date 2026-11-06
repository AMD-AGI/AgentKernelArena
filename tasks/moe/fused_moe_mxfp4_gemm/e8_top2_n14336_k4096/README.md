# MoE expert GEMM, MXFP4 activations and weights, 8 experts, top-2, N 14336, K 4096

One expert projection of a routed mixture-of-experts layer with MXFP4 operands (OCP E2M1 codes with
one E8M0 scale per 32 values along K): every token `t` is multiplied by the weights of each of its
`top_k` selected experts, `c[t, k] = A[t] @ B[topk_ids[t, k]]^T`, accumulated in high precision and
rounded to bfloat16. No routing weight is applied and nothing is reduced over the slots. This is AITER's
Triton `fused_moe_mxfp4` (the kernel vLLM-style Triton MoE stages use for MXFP4 x MXFP4), as exercised
by AITER's `op_tests/triton_tests/moe/test_moe_mx.py`.

## Scenario

- `K = 4096`, `N = 14336`, 8 experts, top-2: the Mixtral-8x7B configuration `(M, 14336, 4096, 8, 2)` of
  the source task's `test_moe_mx.py` shape list (the gate projection of one expert, hidden 4096,
  FFN width 14336). The source task timed 15 unrelated shapes with fixed `M`; this task keeps the one
  production model shape and varies the token count. The other entries of that list are unit-test
  shapes of `test_moe_mx.py` (for example `N = 704, K = 800, E = 3`, `N = 7186`, `E = 128` with top-4, a
  single expert), not a model's layer; in schema v3 each would be its own scenario, and none is
  production-relevant enough to be one, so their unaligned-`N`/`K` and expert-count variants are not
  carried over.
- Var axis: `n_tokens`, 1 to 4096 (decode batches through prefill chunks; the source shape used 64).
  Even one token reads two experts' weights (about 62 MB), estimated near 18 us on MI355X, so the
  decode and prefill sizes 1, 4, 16, 64, 256, 1024, 2048 and 4096 are all timed; 3, 37 and 300 tokens
  are correctness-only boundary cases.

## Interface

`a [n_tokens, 2048]` / `a_scale [n_tokens, 128]` are the already-quantized activations and
`b [8, 14336, 2048]` / `b_scale [8, 14336, 128]` the expert weights: codes packed two per byte along
K (even index in the low nibble) and uint8 E8M0 scales (`2^(s - 127)`), plain row-major. As in the
source task (and the upstream function), the expert-sorted work layout arrives as inputs too: vLLM's
`moe_align_block_size` output for the tile height `align_block` (64 below 256 tokens, 128 from 256 on:
the `BLOCK_SIZE_M` of AITER's gfx950 MX_FP4 MoE table for that token count). `sorted_token_ids` lists
the `(token, slot)` indices `t * top_k + k` grouped by expert, each group padded to a multiple of
`align_block` with the sentinel `n_tokens * top_k`; `expert_ids` gives the expert of each tile (-1 past
the end) and `num_tokens_post_padded` the padded total. The result depends on `topk_ids` alone; a
candidate may use the given layout or build its own. The output `c [n_tokens, 2, 14336]` is bfloat16.
The source call's `topk_weights`, unit per-tensor scales, swizzle flags and tile config are not part of
the interface: the scenario applies no routing weight, and the tiling is the implementation's choice.

## Inputs and reference

Activations and weights are standard-normal bfloat16 quantized with AITER's dynamic MXFP4 rule (the
block maximum rounded to a power of two, up when its mantissa is at least 1.75; scale = that power over
4; round to nearest even E2M1, saturating at 6), as the source harness drew them; the router draws
standard-normal logits and keeps the top-2 softmax entries, so a token's two experts are distinct.
`reference.py` decodes the codes in float64 and multiplies per expert in float64.

Every product of two E2M1 values with E8M0 scales, and every partial sum at these magnitudes, is exact
in float32, so a float32-accumulating kernel computes the exact dot product and its only error is the
final bfloat16 rounding. The comparison is therefore the default `baseline_relative` bound with no
allowance: any correct float32-accumulating implementation, in any accumulation order or split, matches
the baseline's rounding. The source task's bf16 einsum reference with `atol = rtol = 0.1` is replaced.

## Baseline

AITER's `_fused_moe_kernel_mxfp4` and its `fused_moe_mxfp4` launcher (ROCm/aiter `652f2d4b`, MIT; the
source task carried the same code inlined), with AITER's gfx950 MX_FP4 MoE tile table selected by token
count. Changes: the logging `repr=` and the logger are dropped; the launcher is reduced to MXFP4 x MXFP4
without swizzled scales, with unit per-tensor scales and no routing weight; and the activation-scale
load is also masked by the token mask. Upstream loads the scale row of the padding sentinel
(`n_tokens`), one row past the end of `a_scale`; in the source task that read landed in unrelated memory
and only fed masked-out rows, but here `a_scale` is an exactly sized evaluator buffer, so the read is
masked instead. The result of every valid row is unchanged.
