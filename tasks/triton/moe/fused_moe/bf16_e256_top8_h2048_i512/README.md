# Routed-expert MoE, bfloat16, 256 experts, top-8, hidden 2048, intermediate 512, SiLU

The routed-expert computation of one MoE layer, given the router's decisions: for every token and each of its 8
selected experts, `gate/up = x @ w13[e].T`, `a = silu(gate) * up`, `y = (a @ w2[e].T) * routing_weight`, and the
token's output is the sum of its 8 `y` rows, rounded to bfloat16. Routing (gate logits, top-k, softmax) is not part of
the operator; `topk_ids` and `topk_weights` are inputs.

Scenario: the expert layer geometry of the old task this one replaces: 256 experts, top-8, hidden size 2048,
intermediate size 512 (`w13` is `[256, 1024, 2048]`, `w2` is `[256, 2048, 512]`), the MoE configuration of the
Qwen3.5-35B-A3B serving path that SGLang runs through its Triton MoE runner when AITER is disabled. The old task's
128-expert and top-4 variants are different scenarios and not part of this task.

Workloads: token counts from 4 to 16384 (decode batches to batched prefill; the old task swept 16 to 4096). Batches of
1, 3, 100 and 1000 tokens are correctness-only boundary cases. Weights are normal with standard deviation
`1/sqrt(fan_in)` so activations stay O(1); routing selects 8 distinct experts per token from random normal logits with
softmax weights over the selected logits.

Baseline: SGLang's Triton fused-MoE path: `fused_moe_kernel` for the gate/up and down GEMMs with SGLang's default tile
configuration for unquantized weights (small tiles when `tokens <= experts`), `act_and_mul_kernel` for SiLU-and-mul and
`_moe_sum_reduce_kernel` for the top-k sum, all vendored from the pinned SGLang revision with the quantization, LoRA,
TMA, SwiGLU-fusion and PDL branches removed. `moe_align_block_size` is a JIT CUDA kernel in SGLang and is replaced by a
Triton kernel written for this task (`align_kernels.py`, atomic counters; the order of tokens inside an expert is
unspecified, which does not change any result), so the baseline contains no PyTorch compute. The old task ran the SiLU
and the top-k sum as PyTorch float32 operations outside the kernels; they are Triton kernels here. Intermediates are
bfloat16.

Bound: the default `baseline_relative` bound against the float64 reference (per-expert gather/GEMM/index-add in
float64, independent of the baseline).

The kernels of this baseline have not been run on a GPU as part of authoring (CPU-only pre-check); the QA audit on
MI355X is the first execution.
