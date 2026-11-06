"""Expert weights and router decisions of Mixtral-8x7B.

The expert weights are normal draws with standard deviation 1 / sqrt(fan-in), so that each projection has
unit-order outputs. They are drawn one expert at a time directly into bfloat16, which keeps input preparation from
holding a float32 copy of the whole 2.8 GB weight stack.

Each token picks the top 2 of 8 experts by its gate logits. The routing weights are the softmax of the gate logits
restricted to the two selected experts (Mixtral's softmax over all experts followed by renormalization of the top
2), so they sum to one per token.
"""
import torch

NUM_EXPERTS = 8
TOP_K = 2


def _expert_weights(shape, std, generator, device):
    """Normal(0, std) weights of shape [experts, rows, cols] in bfloat16, drawn one expert at a time."""
    weights = torch.empty(shape, dtype=torch.bfloat16, device=device)
    for expert in range(shape[0]):
        weights[expert] = torch.randn(shape[1:], generator=generator, device=device).mul_(std)
    return weights


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    num_experts, hidden, inter = axes["num_experts"], axes["hidden_size"], axes["intermediate_size"]
    assert num_experts == NUM_EXPERTS and axes["top_k"] == TOP_K
    w13 = _expert_weights((num_experts, 2 * inter, hidden), hidden ** -0.5, generator, device)
    w2 = _expert_weights((num_experts, hidden, inter), inter ** -0.5, generator, device)
    logits = torch.randn(axes["num_tokens"], NUM_EXPERTS, generator=generator, device=device)
    values, ids = logits.topk(TOP_K, dim=-1)
    return {"w13": w13, "w2": w2, "topk_ids": ids.to(torch.int32).contiguous(),
            "topk_weights": torch.softmax(values, dim=-1).contiguous()}
