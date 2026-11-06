# Fused MoE, DeepSeek-V3 on tensor parallelism 8 with the fused shared expert, bfloat16

This is the routed-expert block of a DeepSeek-V3 MoE layer on one rank of tensor parallelism 8. It has 256
routed experts plus the shared expert, fused in as expert 256. The hidden size is 7168 and the expert
intermediate width is 2048 / 8 = 256. Each token runs through its 9 experts (8 routed and the shared one):
`out[t] = sum_k topk_weights[t, k] * (silu(x W1[e]^T) * (x W3[e]^T)) W2[e]^T` with `e = topk_ids[t, k]`. `w13`
stacks the gate projection (`W1`, the first 256 rows) and the up projection (`W3`, the last 256 rows) of each
expert. The router decisions are inputs, so the operator starts from the given `topk_ids` and `topk_weights`.
Sorting the tokens by expert is part of the operator and is timed.

## Origin

The source task targets the generic, unquantized two-stage fused MoE of the AMD runtime: `aiter.fused_moe` with
`QuantType.No`. That path runs a grouped gate/up GEMM with a SiLU-and-multiply epilogue, keeps the intermediate
activation in bfloat16, then runs the grouped down GEMM with the weighted top-k combine. It dispatches to Composable
Kernel and assembly MoE kernels, which cannot be vendored as source. The baseline is therefore AITER's Triton MoE
implementation of the same two stages (`aiter/ops/triton/moe`). It runs in the order of a Triton fused-MoE
runner:

1. `moe_align_block_size` (four Triton kernels) sorts the token-slot pairs by expert and pads each expert's run to
   the GEMM tile height.
2. `fused_moe_silu` computes `silu(gate) * up` for each routed slot into a bfloat16 intermediate.
3. `fused_moe` computes the down projection of each routed slot, multiplies it by the slot's routing weight and
   writes it in bfloat16.
4. `reduce_grouped` sums the 9 slot rows of each token in float32 and rounds the result to bfloat16.

The tile configurations come from AITER's tuned gfx950 bfloat16 MoE table and are chosen by token count, as in
AITER's `get_optimal_moe_config`. The vendored kernels are verbatim apart from two changes: `_reduce_grouped` loses
its gfx1250-only tensor-descriptor loads, and `_swiglu`'s parameter `input` is renamed because the candidate rules
reserve that name. Only the unquantized, non-persistent Triton launch paths are kept (see the provenance in
`task.yaml`). The baseline may be slower than the CK and assembly production kernels; it is still the task's
baseline. In particular, AITER's Triton token sort runs one program per expert, and two of its stages loop
serially over all 257 experts. That gives it a fixed cost in every case, which is largest relative to the GEMMs
in the decode cases.

## Scenario and workloads

`inputs.py` draws the weights from a normal distribution with standard deviation `1 / sqrt(fan-in)`, so every
projection has unit-order outputs. The weights are drawn one expert at a time directly in bfloat16, which avoids
up to 11 GB of float32 temporaries for the whole weight stack. Activations are standard normal. The router inputs follow
DeepSeek-V3's gate. Group-limited, bias-corrected sigmoid scores select 8 of the 256 routed experts (8 groups of
32, best 4 groups), and their weights are renormalized to sum to one. As in SGLang's fused shared experts, the
shared expert 256 takes the last slot with weight `1 / routed_scaling_factor` (2.5). The correction bias skews
the load across experts, as in the real model: 64 tokens touch about 184 experts, 256 tokens about 234 and 1024
tokens about 251.

The token counts cover decode batches (1 to 256 tokens) and prefill chunks (1024 to 16384 tokens, the chunked
prefill sizes SGLang uses). Even one decode token reads the weights of 9 experts (about 99 MB), so every
performance case is above the 10 µs timing floor. Odd and unaligned batch sizes are correctness-only.

The source task timed two 32-token cases of the production operator. Its DeepSeek-V3 shape is this scenario, now
swept over production token counts, with the 32-token batch kept as one of the cases. Its second shape (32
experts, top 5, hidden size 4096) is not a named production configuration and is not carried over.

## Comparison bound: `baseline_relative` with `k: 5`

The baseline rounds three intermediates to bfloat16 before the float32 top-k sum: the gate/up accumulator, the SiLU
product and each slot's down projection. Its error is the sum of those roundings. A correct implementation that
rounds fewer intermediates, or rounds them at other points, has a different error, not a larger one. Examples are
a fused down projection and combine that keeps the slots in float32, or an epilogue that applies SiLU to the
float32 accumulator.

A CPU emulation of the baseline's roundings (SQNR 48.2 dB against the float64 reference) measured two such
variants against the bound, as peak multiples of it. Keeping the slot outputs in float32 (49.1 dB) and computing
the bfloat16 SiLU product from the unrounded gate/up accumulator (50.8 dB) gave the peaks below. Each range covers
both variants and two emulations with independent input draws (the 256-token row comes from one draw), each
comparing every token of the case under the evaluator's bound.

| Tokens | `k: 2` | `k: 3` | `k: 4` | `k: 5` |
| --- | --- | --- | --- | --- |
| 256 | 1.12 to 1.53 | 0.89 to 1.05 | 0.73 to 0.79 | 0.62 to 0.69 |
| 2048 | 1.26 to 1.58 | 0.96 to 1.17 | 0.80 to 0.94 | 0.69 to 0.83 |
| 16384 | 1.41 to 1.90 | 0.99 to 1.29 | 0.85 to 0.98 | 0.75 to 0.84 |

A multiple above 1 means the variant fails. With the default `k: 2`, both variants fail on about 20 to 110 elements
per million; even the most accurate of these designs, which does both (52.6 dB), peaks at 1.22. `k: 4` leaves only a
2% margin on the largest case, where the extreme tails are longest; `k: 5` passes every variant there with a 16%
margin, and the mean-error (bias) check passes them too. The bound stays strict. An intermediate quantized to fp8
with per-row scales (32.3 dB) violates it on 31% of the elements, a top-k sum rounded to bfloat16 after every slot
(46.7 dB) on about 10 elements per million, and the evaluator's 16-ulp shift and fp8-output controls are far
outside it.
