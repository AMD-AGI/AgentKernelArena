# Routed-expert MoE, bfloat16, 128 experts, top-8, hidden 2816, intermediate 352, GELU (tanh)

The routed-expert computation of one MoE layer, given the router's decisions: for every token and each of its 8
selected experts, `gate/up = x @ w13[e].T`, `a = gelu_tanh(gate) * up`, `y = (a @ w2[e].T) * routing_weight`, and the
token's output is the sum of its 8 `y` rows, rounded to bfloat16. Routing itself (gate logits, top-k, softmax) is not
part of the operator; `topk_ids` and `topk_weights` are inputs.

Scenario: the per-rank expert tensors of Gemma-4-26B-A4B served with tensor parallelism 2 and expert parallelism 1
(128 experts, top-8, hidden 2816, per-rank intermediate 352, so `w13` is `[128, 704, 2816]` and `w2` is
`[128, 2816, 352]`, GELU tanh approximation). ROCm AITER's unquantized MoE backend does not support the GELU-tanh
activation, so vLLM runs this layer through its Triton fused-MoE path on MI355X.

Workloads: token counts of the model's decode (64), chunked-prefill (1080, 7218) steps plus a sweep from 4 to 16384
tokens. Batches of 1, 3, 100 and 1000 tokens are correctness-only boundary cases. Weights are normal with standard
deviation `1/sqrt(fan_in)` so activations stay O(1); routing selects 8 distinct experts per token from random normal
logits with softmax weights over the selected logits.

Baseline: vLLM's Triton path, the sequence of `fused_experts_impl` (align, gate/up GEMM, activation, down GEMM with the
routing weight, top-k sum), using vLLM's `fused_moe_kernel` for both GEMMs with the default tile configuration for
unquantized bfloat16 on ROCm (no tuned configuration file exists for E=128, N=352 on MI355X). Three steps are CUDA ops
in vLLM and are replaced here by Triton kernels written for this task, so the baseline contains no PyTorch compute:
`moe_align_block_size` (`align_kernels.py`, atomic counters; the order of tokens inside an expert is unspecified, which
does not change any result), `gelu_tanh_and_mul` and `moe_sum` (`moe_ops.py`, activation rounded to bfloat16 before the
multiply, top-k sum in float32). The intermediates are bfloat16, as in vLLM.

Bound: the default `baseline_relative` bound against the float64 reference (per-expert gather/GEMM/index-add in
float64, independent of the baseline). Every performance case is expected to exceed the 10 us floor because the
baseline launches about a dozen kernels and reads the weights of every selected expert.

The kernels of this baseline have not been run on a GPU as part of authoring (CPU-only pre-check); the QA audit on
MI355X is the first execution.
