"""Expert weights and router decisions of DeepSeek-V3 with its shared expert fused into the routed experts.

The expert weights are normal draws with standard deviation 1 / sqrt(fan-in), so that each projection has
unit-order outputs. They are drawn one expert at a time directly into bfloat16, which keeps input preparation from
holding a float32 copy of the whole 2.8 GB weight stack.

DeepSeek-V3 routes each token to 8 of 256 experts with group-limited, bias-corrected sigmoid gating: the experts
form 8 groups of 32, the 4 groups with the highest sum of their two best bias-corrected scores are kept, and the
8 best bias-corrected scores inside them are selected. The routing weights are the selected sigmoid scores (without
the bias), renormalized to sum to one. As in SGLang's fused shared experts, the shared expert is the extra expert
256 in the last slot of every token, weighted by 1 / routed_scaling_factor relative to the routed experts.
"""
import torch

NUM_ROUTED_EXPERTS = 256
NUM_EXPERT_GROUPS = 8
TOPK_GROUPS = 4
ROUTED_TOP_K = 8
ROUTED_SCALING_FACTOR = 2.5
CORRECTION_BIAS_STD = 0.05


def _expert_weights(shape, std, generator, device):
    """Normal(0, std) weights of shape [experts, rows, cols] in bfloat16, drawn one expert at a time."""
    weights = torch.empty(shape, dtype=torch.bfloat16, device=device)
    for expert in range(shape[0]):
        weights[expert] = torch.randn(shape[1:], generator=generator, device=device).mul_(std)
    return weights


def _router(num_tokens, generator, device):
    logits = torch.randn(num_tokens, NUM_ROUTED_EXPERTS, generator=generator, device=device)
    bias = CORRECTION_BIAS_STD * torch.randn(NUM_ROUTED_EXPERTS, generator=generator, device=device)
    scores = logits.sigmoid()
    choice = scores + bias
    group_scores = choice.view(num_tokens, NUM_EXPERT_GROUPS, -1).topk(2, dim=-1).values.sum(dim=-1)
    group_idx = group_scores.topk(TOPK_GROUPS, dim=-1).indices
    group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1.0).bool()
    expert_mask = group_mask.repeat_interleave(NUM_ROUTED_EXPERTS // NUM_EXPERT_GROUPS, dim=1)
    ids = choice.masked_fill(~expert_mask, float("-inf")).topk(ROUTED_TOP_K, dim=-1).indices
    weights = scores.gather(1, ids)
    routed_sum = weights.sum(dim=-1, keepdim=True)
    weights = torch.cat([weights, routed_sum / ROUTED_SCALING_FACTOR], dim=-1) / routed_sum
    shared = torch.full((num_tokens, 1), NUM_ROUTED_EXPERTS, dtype=ids.dtype, device=device)
    ids = torch.cat([ids, shared], dim=-1)
    return ids.to(torch.int32).contiguous(), weights.float().contiguous()


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    num_experts, hidden, inter = axes["num_experts"], axes["hidden_size"], axes["intermediate_size"]
    assert num_experts == NUM_ROUTED_EXPERTS + 1 and axes["top_k"] == ROUTED_TOP_K + 1
    w13 = _expert_weights((num_experts, 2 * inter, hidden), hidden ** -0.5, generator, device)
    w2 = _expert_weights((num_experts, hidden, inter), inter ** -0.5, generator, device)
    topk_ids, topk_weights = _router(axes["num_tokens"], generator, device)
    return {"w13": w13, "w2": w2, "topk_ids": topk_ids, "topk_weights": topk_weights}
