# Routed experts with stage-1 router weighting (tkw1), bfloat16, 128 experts, top-1, hidden 5120, intermediate 1024

The routed-expert block of a Llama 4 Maverick layer. Each token has one selected expert `e = topk_ids[t, 0]` and a
routing weight `r = topk_weights[t, 0]`. The router weight is applied in the first stage ("top-k weight on stage 1",
tkw1): it scales both the gate and the up projection before the activation,

```text
h    = hidden_states[t] @ w13[e].T          # [2048]: gate = h[:1024], up = h[1024:]
a    = silu(r * gate) * (r * up)
out[t] = a @ w2[e].T                         # rounded to bfloat16; r is not applied again
```

This is the semantics of AITER's `fused_moe(..., doweight_stage1=True)` (`asm_moe_tkw1`, reference
`torch_moe_tkw1`) and of vLLM's `apply_router_weight_on_input` used for Llama 4 (the router weight multiplies the
first GEMM's accumulator). Because silu is not linear, applying the weight after the down projection instead gives a
different result, which the comparison rejects.

**Scenario.** 128 routed experts, top-1, hidden size 5120, expert intermediate size 8192 split over tensor parallelism
8 (1024 per rank), bfloat16 weights and activations, float32 routing weights, int32 expert ids. `w13` holds the gate
rows (0..1023) followed by the up rows (1024..2047) of each expert, as in AITER and vLLM. Only the token count varies.

**Inputs.** Activations are standard normal. `inputs.py` draws the expert weights normal with standard deviation
1/sqrt(K) of their reduction (1/sqrt(5120) for `w13`, 1/sqrt(1024) for `w2`), rounded to bfloat16, which keeps the
gate, the activation and the output of order one. It draws them a few experts at a time because the bank is 4 GB per
draw. It draws the router decisions the way Llama 4 makes them: the expert with the largest router logit, scored with
the sigmoid of that logit and not renormalized. A per-token logit offset spreads the selected score over (0, 1)
instead of clustering near one, so the placement of the weight is exercised. A per-expert popularity bias of the
logits (standard deviation 0.25, drawn anew with every input draw) makes the expert loads uneven, as a trained
router's are: at 16384 tokens the most loaded expert receives about 350 to 800 tokens (3 to 6 times the mean) and the
least loaded only a few. With uniform routing no expert would receive more than about 165 tokens in any case, so no
expert's segment would ever span more than one of the baseline's 256-row tiles; the source task's harness checked such
occupancies explicitly.

**Workloads.** Token counts of a serving step: decode batches 16 to 256, chunked prefill steps (1024 to 4096) and
batched prefill (8192, 16384); the performance cases span 1024x. Even a 16-token step reads about 15 experts' weights
(31.5 MB each), so every performance case should exceed 10 us. Batches of 1 and 3 tokens and unaligned counts (33,
100, 1000, 5000) are correctness-only.

**Baseline.** The production function is a CK/assembly kernel, which cannot be vendored, so the baseline is the same
computation built from AITER's own Triton MoE kernels (see `provenance` in `task.yaml`; `baseline/LICENSE` is the
AITER MIT license):

1. `moe_align_block_size_triton` groups the token slots by expert, padded to the tile height;
2. `fused_moe_silu` with `mul_routed_weight=True`: gate/up GEMM with float32 accumulation, both projections multiplied
   by the router weight in float32 and rounded to bfloat16, then `silu(gate) * up` rounded to bfloat16;
3. `fused_moe` with `mul_routed_weight=False`: the down GEMM, written straight into the output (top_k is 1, so no
   top-k sum is needed).

The tile configuration is AITER's tuned gfx950 table for unquantized MoE (`gfx950-MOE-DEFAULT.json`, selected by token
count: below 256, below 1024, otherwise), used for every step. The device kernels are unchanged; the launchers are
reduced to their unquantized, non-persistent path. This Triton baseline may be slower than the production CK/assembly
kernel; it is still the task's baseline. Its large-batch tile (256 rows) pads each expert's few tokens heavily at 1024
to 4096 tokens, so the baseline time is not monotone in the token count.

**Comparison bound.** `out` uses `baseline_relative` with `k: 4` and `max_violation_fraction: 1.0e-5` instead of the
defaults (2, 0). The baseline's error is dominated by its two bfloat16 intermediate roundings (the scaled gate/up and
the activation), so a correct implementation that rounds at different points or reduces in a different order has
errors of the same size at different elements. A CPU emulation of the baseline's rounding against this reference
(16 experts, real dimensions, 8192 tokens: 42M output elements) gave these results:

| Implementation | default bound (k 2) | this bound |
| --- | --- | --- |
| split-K reordering of both GEMMs | fails, peak 1.06x | passes, peak 0.70x |
| gate/up kept in float32 (more accurate) | fails, peak 1.46x | passes, peak 0.73x |
| activation kept in float32 (more accurate) | fails, peak 1.33x | passes, peak 0.83x |
| gate/up rounded to bfloat16 before the weight is applied (extra rounding) | fails | fails, 1.1e-4 of elements |
| activations pre-scaled by the weight in bfloat16 before the GEMM (extra rounding) | fails | fails, 5.6e-5 of elements |
| output scaled by 1 + 2^-8 (0.4 %) | fails | fails, 4.4e-5 of elements |
| output scaled by 1 + 2^-6 (1.6 %) / the control suite's 16-ulp scaling | fails | fails, 19 % / 80 % of elements |
| float8 rounding of the output | fails | fails, 56 % of elements |
| weight applied after the down projection (tkw2), or only to the gate | fails | fails, 98 % of elements |
| one token row computed as tkw2 | fails | fails |

The violation fraction admits about 840 elements of the largest case, less than one token row (5120 elements), so a
single wrongly computed token is still rejected. Implementations that add a bfloat16 rounding the baseline does not
have (the two "extra rounding" rows) are less accurate than the baseline and are rejected. Pre-scaling the
activations by the router weight in bfloat16, as the Hugging Face model and vLLM's modular MoE path do, is such an
extra rounding: apply the weight to the float32 accumulator instead, as AITER's and vLLM's fused Triton kernels do.

**Relation to the old task.** The source task timed an FP8 per-token variant (weights and activations quantized to
float8 inside every call, cases E = 16/128, top-1/top-2 at 128 tokens). This scenario is the bfloat16 Llama 4 Maverick
layer with production token counts; the FP8 per-token variant is not reproduced here. Of the old cases, only the top-1
ones match a production model: Llama 4, the model that applies the router weight on the input, routes top-1 (vLLM's
modular MoE path implements that weighting for top-1 only). The 16-expert top-1 case is the Llama 4 Scout layer, a
separate scenario (a sibling task with `num_experts` 16) that this package does not cover. The old harness's checks of
experts holding more tokens than one tile are kept through the uneven routing described above.

**Memory.** The weights are 4 GB per draw; with two draws, the working copy and the baseline outputs, a case needs
roughly 20 GB of device memory. The float64 reference loops over the 128 experts.
