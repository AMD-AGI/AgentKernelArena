"""High-precision routed-experts MoE reference used only for correctness."""
import math

import torch


def reference(*, hidden_states, w13, w2, topk_weights, topk_ids):
    num_tokens, hidden = hidden_states.shape
    num_experts, gate_up, _ = w13.shape
    inter = gate_up // 2
    out = torch.zeros((num_tokens, hidden), dtype=torch.float64, device=hidden_states.device)
    weights = topk_weights.double()
    for expert in range(num_experts):
        token, slot = (topk_ids == expert).nonzero(as_tuple=True)
        if token.numel() == 0:
            continue
        x = hidden_states[token].double()
        h = x @ w13[expert].double().t()
        gate, up = h[:, :inter], h[:, inter:]
        act = 0.5 * gate * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (gate + 0.044715 * gate ** 3))) * up
        y = (act @ w2[expert].double().t()) * weights[token, slot].unsqueeze(1)
        out.index_add_(0, token, y)
    return out
