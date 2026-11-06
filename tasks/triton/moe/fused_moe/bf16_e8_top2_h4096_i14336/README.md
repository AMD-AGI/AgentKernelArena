# Fused MoE, Mixtral-8x7B, bfloat16

This is the routed-expert block of a Mixtral-8x7B MoE layer without tensor parallelism: 8 experts, 2 per token,
hidden size 4096 and expert intermediate width 14336. Each token runs through its 2 experts:
`out[t] = sum_k topk_weights[t, k] * (silu(x W1[e]^T) * (x W3[e]^T)) W2[e]^T` with `e = topk_ids[t, k]`. `w13`
stacks the gate projection (`W1`, the first 14336 rows) and the up projection (`W3`, the last 14336 rows) of each
expert. The router decisions are inputs, so the operator starts from the given `topk_ids` and `topk_weights`.
Sorting the tokens by expert is part of the operator and is timed.

## Origin

The two source tasks each timed one expert GEMM of this layer. One was AITER's Triton `fused_moe` kernel with the
routing-weight multiply (a standalone copy of `aiter.ops.triton.moe`). The other was the ROCm benchmark's
`moe_gemm` kernel on the Mixtral-8x7B shapes (N = 14336, K = 4096). This task times the whole routed-expert layer
instead, the same operator as the other `fused_moe` scenarios: both expert GEMMs, the SiLU-and-multiply activation
and the weighted top-k sum. The single GEMM stays at the core of the baseline.

The baseline is AITER's Triton MoE implementation (`aiter/ops/triton/moe`). It runs in the order of a Triton
fused-MoE runner:

1. `moe_align_block_size` (four Triton kernels) sorts the token-slot pairs by expert and pads each expert's run to
   the GEMM tile height.
2. `fused_moe_silu` computes `silu(gate) * up` for each routed slot into a bfloat16 intermediate.
3. `fused_moe` (the current revision of the first source task's kernel) computes the down projection of each
   routed slot, multiplies it by the slot's routing weight and writes it in bfloat16.
4. `reduce_grouped` sums the 2 slot rows of each token in float32 and rounds the result to bfloat16.

The tile configurations come from AITER's tuned gfx950 bfloat16 MoE table and are chosen by token count, as in
AITER's `get_optimal_moe_config`. The vendored kernels are verbatim apart from two changes: `_reduce_grouped` loses
its gfx1250-only tensor-descriptor loads, and `_swiglu`'s parameter `input` is renamed because the candidate rules
reserve that name. Only the unquantized, non-persistent Triton launch paths are kept (see the provenance in
`task.yaml`). AITER's Triton token sort runs one program per expert. With only 8 experts, its counting and
scatter stages each loop serially over an eighth of the routed slots per program (2048 slots at 8192 tokens), so
the sort takes a noticeable share of the large prefill cases. It is still the task's baseline.

## Scenario and workloads

`inputs.py` draws the weights from a normal distribution with standard deviation `1 / sqrt(fan-in)`, so every
projection has unit-order outputs. The weights are drawn one expert at a time directly in bfloat16, which avoids
up to 11 GB of float32 temporaries for the whole weight stack. Activations are standard normal. The router inputs
follow Mixtral: the top 2 of 8 normal gate logits, weighted by the softmax over the two selected logits.

The token counts cover decode batches (1 to 256 tokens) and prefill chunks (1024 to 8192 tokens, vLLM's default
chunked-prefill budget). One decode token already reads two experts (about 700 MB of weights), so every
performance case is above the 10 µs timing floor. Odd and unaligned batch sizes are correctness-only. The source
tasks' token counts on this shape (64, 256 and 1024) are among the cases, and both settings of their
routing-weight switch are exercised: the gate/up GEMM runs without the routing weight, the down GEMM with it.

The source tasks also tested other variants of the single GEMM, which this bfloat16 scenario does not carry: the
ROCm benchmark's float16 inputs and its per-tensor fp8 (W8A8) and int8 (W8A16, W8A8) quantized modes, and both
tasks' small synthetic shapes (4 to 32 experts, top 1 to 4, K down to 1). A scenario fixes its dtypes and
constant dimensions, Mixtral-8x7B is served in bfloat16, and a quantized MoE GEMM needs its own scales, reference
and bound, so those variants belong in sibling scenarios rather than in this task's cases.

## Comparison bound: `baseline_relative` with `k: 5`

The baseline rounds three intermediates to bfloat16 before the float32 top-k sum: the gate/up accumulator, the SiLU
product and each slot's down projection. Its error is the sum of those roundings. A correct implementation that
rounds fewer intermediates, or rounds them at other points, has a different error, not a larger one. Examples are
a fused down projection and combine that keeps the slots in float32, or an epilogue that applies SiLU to the
float32 accumulator.

A CPU emulation of the baseline's roundings (SQNR 48.1 dB against the float64 reference) measured two such
variants against the bound, as peak multiples of it. Keeping the slot outputs in float32 (49.1 dB) and computing
the bfloat16 SiLU product from the unrounded gate/up accumulator (50.6 dB) gave the peaks below. Each range covers
both variants and two emulations with independent input draws (the 256-token row comes from one draw), each
comparing every token of the case under the evaluator's bound.

| Tokens | `k: 2` | `k: 3` | `k: 4` | `k: 5` |
| --- | --- | --- | --- | --- |
| 256 | 1.04 to 1.43 | 0.84 to 0.97 | 0.71 to 0.75 | 0.61 to 0.65 |
| 8192 | 1.23 to 1.87 | 0.94 to 1.28 | 0.79 to 0.98 | 0.69 to 0.85 |

A multiple above 1 means the variant fails. With the default `k: 2`, both variants fail on 9 to 81 elements per
million; even the most accurate of these designs, which does both (52.6 dB), peaks at 1.08. `k: 4` leaves only a
2% margin on the largest case; `k: 5` passes every variant there with a 15% margin. The bound stays strict. An
intermediate quantized to fp8 with per-row scales (31.5 dB) violates it on 35% of the elements, and the
evaluator's 16-ulp shift and fp8-output controls are far outside it.
