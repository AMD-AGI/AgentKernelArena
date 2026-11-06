"""Per-expert token counts of a DeepGEMM masked MoE layer.

Expert 0 is full (count == T), expert 1 is empty, the others hold between T/2 and T valid tokens, as after a
low-latency dispatch with unevenly loaded experts. Rows beyond a count carry arbitrary activations.
"""
import torch


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    n_experts, capacity = axes["num_experts"], axes["max_tokens"]
    low = capacity // 2
    counts = torch.randint(low, capacity + 1, (n_experts,), generator=generator, device=device, dtype=torch.int64)
    counts[0] = capacity
    if n_experts > 1:
        counts[1] = 0
    return {"tokens_per_expert": counts.to(torch.int32)}
