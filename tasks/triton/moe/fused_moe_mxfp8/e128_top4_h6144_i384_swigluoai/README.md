# Routed-expert MoE with MXFP8 weights, 128 experts, top-4, hidden 6144, intermediate 384, swigluoai (GPT-OSS SwiGLU)

The routed-expert computation of one MiniMax-M3 MoE layer on one tensor-parallel rank (TP=8: intermediate
3072 / 8 = 384), given the router's decisions. The activations are quantized to MXFP8 (FP8-E4M3 values, one
UE8M0 scale per 1x32 block along the reduction axis); for every token and each of its 4 selected experts,
`g = x_q @ w13[e].T` (gate = first half, up = second half, bfloat16), the swigluoai activation
`min(gate, 7) * sigmoid(1.702 * min(gate, 7)) * (clamp(up, -7, 7) + 1)`, a second MXFP8 quantization, and
`y = (a_q @ w2[e].T) * routing_weight`; the token's output is the sum of its 4 `y` rows, rounded to bfloat16.
Routing (gate logits, top-k, softmax) is not part of the operator; `topk_ids` and `topk_weights` are inputs.
The task covers the whole pipeline from bfloat16 activations to the bfloat16 output, including expert
alignment, both quantizations and the activation, which the old task kept outside its timed region
(it timed only the two grouped GEMM launches).

Scenario: SGLang's native MXFP8 MoE for MiniMax-M3 on gfx950 (`fused_moe_mxfp8_native`), 128 experts,
top-4, hidden 6144, per-rank intermediate 384, `alpha = 1.702`, `beta = 1`, `limit = 7`. `w13` is
`[128, 768, 6144]`, `w2` is `[128, 6144, 384]` (both float8_e4m3fn) with `uint8` UE8M0 scales of shape
`[128, 768, 192]` and `[128, 6144, 12]`.

## Workloads

Token counts from 4 to 16384: decode batches, chunked prefill and large prefill batches. The old task's
cases (1, 64 and 16384 tokens) are all kept; batch 1 (about 28 MB of expert weights read) is
correctness-only because the pipeline is launch-bound there, as are the unaligned boundary cases 3, 100
and 1000. The old task capped correctness at 64 tokens; here every case, including 16384 tokens, is checked
against the float64 reference.

Inputs (`inputs.py`): activations are `N(0, 0.5^2)`; expert weights are `N(0, 0.1^2)` in bfloat16 quantized to
MXFP8 with the operator's quantizer; routing picks 4 distinct experts per token from random normal logits and
its weights are the softmax over the selected logits (they sum to one; the old task used the un-renormalized
softmax of all 128 experts).

## Baseline

SGLang's native MXFP8 path (sgl-project/sglang at `887c401e151e2d1b082d74ef8769a23eef5c465c`; the old task
pinned an earlier revision and an older file path), all Triton: the grouped `tl.dot_scaled` GEMM kernel
(`_mxfp8_grouped_gemm_kernel` / `_grouped_gemm_mxfp8`, GEMM1 gathers activations by `token // top_k`, GEMM2
applies the routing weight), the MXFP8 quantization kernel, the swigluoai kernel (default two-kernel
activation path: bfloat16 activation, then a separate quantization; the opt-in fused swiglu+quant kernel is not
used) and the top-k combine kernel, sequenced as `fused_moe_mxfp8_native` does. Removed: expert-parallel
`expert_map` and `-1` filtering, the `no_combine` branch, PyTorch fallbacks, environment flags.

`moe_align_block_size` is a JIT-compiled CUDA kernel in SGLang; it is replaced by the Triton kernel written for
this repository for the bfloat16 `fused_moe` task (`align_kernels.py`, atomic counters; the order of routes
inside an expert is unspecified, which does not change any result). Upstream's final reduction uses
`g2.view(T, top_k, H).sum(1)` in PyTorch by default; the baseline uses upstream's combine kernel instead, so it
contains no PyTorch compute. The grouped GEMM outputs are zero-filled as upstream does.

## Reference and comparison

`reference.py` runs the matrix products in float64 over exactly dequantized MXFP8 operands, one expert at a time,
independent of the baseline. The operator's rounding steps are semantics, not errors, and are reproduced: the
float32 quantizer (scale exponent `ceil(log2(amax / 448)) + 127`, clamp to +-448, round-to-nearest-even to
float8_e4m3fn) for the activations and for the SwiGLU output, and bfloat16 rounding of GEMM1's output and of
the activation. Without these the 8-bit quantizations would put the baseline's SQNR against an unquantized
reference near the 25 dB sanity limit.

Both bfloat16 roundings decide which e4m3 codes the activation gets, so a candidate must apply them, also when it
fuses the SwiGLU and the quantization. Keeping the activation in float32 through the quantizer (upstream's opt-in
fused SwiGLU-quantization kernel) changes 2.9 % of the activation codes and puts the output at 36 dB SQNR against
the reference, about as far as rounding the output to fp8 (31.5 dB); applying the SwiGLU to the float32 GEMM1
accumulator changes 4.4 % of them. Both fail the bound below on 14 to 24 % of the elements at 4096 tokens.

### Bound: `baseline_relative`, `k: 8`, `max_violation_fraction: 0.005`

Because every rounding is reproduced, an implementation with float32 accumulation matches the reference except
where a float32 difference lands on a rounding boundary. Most such differences are harmless, but now and then a
bfloat16 activation sits on an e4m3 rounding midpoint in one implementation and one bfloat16 step away in another:
the two codes are then one e4m3 step (6 to 12 % of that element) apart, and the difference is multiplied by a
`w2` column and spread over all 6144 outputs of the token. The baseline's own error is only the bfloat16 output
rounding in almost every row, so the default bound (`k` 2, no violations) rejects such a row. In the MI355X QA
audit, the reference evaluated in float32 (a correct float32 implementation) failed the default bound on 3 of the
11 cases: 2615 elements (0.043 %, 7.8x the bound) at 1000 tokens, 2 elements at 4096 and 606 at 16384 tokens.

A CPU emulation at the real sizes (24576 tokens in six input draws; baseline emulated with float32 GEMMs over
K blocks of 128 as its `tl.dot_scaled` loop does and float32 SwiGLU) measured:

- The baseline's bfloat16 GEMM1 output differs from the reference in 6.7e-5 of the elements, its bfloat16
  activation in 1.0e-4 and its activation codes in 4.6e-6 (one route in about 570). Other float32 orders (one
  float32 GEMM per product as PyTorch computes it, K blocks of 256 in reverse order, a SwiGLU through `exp2`)
  have the same rates. Against the emulated baseline, the GEMM-order variants fail the default bound on 4 to 7 of
  the 24576 token rows (peak 3.9x), so they fail 3 to 5 of the six 4096-token groups; under the bound above no
  group of any case size fails (largest violation fraction 1.6e-4, in a single-token row).
- One forced e4m3 step on a random activation element of a random route violates the default bound in 27 % of
  the rows (up to 73 % of a row's elements); at `k` 8 in 9.5 % of the rows (up to 36 %). On the largest element of
  the heaviest route it violates `k` 8 on 22 % of the row's elements (median; at most 49 %). The 0.5 % allowance
  absorbs one such step from 100 tokens on and the few steps the large cases collect; in the cases of 1 to 16
  tokens no permitted allowance can absorb a damaging step.
- Drawing independent steps with that random-element severity at 2e-4 per route (the rate the emulated variants
  imply) or 1e-3 per route (pessimistic), a correct candidate fails some case with a probability of about 100 %
  under the default bound, 0.2 % or 0.9 % with `k` 4 and 1 %, and 0.09 % or 0.42 % with `k` 8 and 0.5 %.
- Under `k` 8 and 0.5 %, the fp8-rounded output control (at least 8.5 % of the elements of any single token,
  34 % from 16 tokens) and the 16-ulp shift control (at least 47 %) fail every case, down to every single-token
  row. From 4 tokens on, every case also rejects the activation kept in float32 through the quantizer (24 % at 4096
  tokens), the SwiGLU applied to the float32 accumulator (14 %), an activation that is not requantized (51 %),
  hidden states that are not quantized (72 %) and a 1 % output gain error (2 %, four times the allowance); of the
  single-token rows, 94 to 100 % reject them. A gain error of 0.5 % (about one bfloat16 step) is not rejected.

## Risks

- The MI355X QA audit passed initial validation, the timing floor and every control with the default bound; the
  bound above changes only the comparison of `out`. The emulation's rates are CPU float32 rates; the audit's
  float32 reference failed the default bound at similar rates, and its failures are within the new allowance
  (largest 0.043 % at 1000 tokens).
- A correct candidate whose float32 rounding differs from the baseline can still fail one of the 1 to 16 token
  cases when one of its activation codes lands one e4m3 step from the baseline's on a large element (estimated
  0.1 to 0.4 % of such candidates).
- Inputs are about 0.9 GB of weights and the fp32 route buffer of the 16384-token case is 1.6 GB.
