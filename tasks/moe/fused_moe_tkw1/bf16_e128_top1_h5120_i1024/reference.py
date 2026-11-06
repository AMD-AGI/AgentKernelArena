"""High-precision reference for routed experts with the router weight applied in stage 1."""
import torch


def reference(*, hidden_states, w13, w2, topk_weights, topk_ids):
    num_tokens, hidden = hidden_states.shape
    num_experts, gate_up, _ = w13.shape
    inter = gate_up // 2
    out = torch.zeros((num_tokens, hidden), dtype=torch.float64, device=hidden_states.device)
    for expert in range(num_experts):
        token, slot = (topk_ids == expert).nonzero(as_tuple=True)
        if token.numel() == 0:
            continue
        x = hidden_states[token].double()
        weight = topk_weights[token, slot].double().unsqueeze(1)
        h = x @ w13[expert].double().t()
        gate = h[:, :inter] * weight
        up = h[:, inter:] * weight
        act = gate * torch.sigmoid(gate) * up
        out.index_add_(0, token, act @ w2[expert].double().t())
    return out
