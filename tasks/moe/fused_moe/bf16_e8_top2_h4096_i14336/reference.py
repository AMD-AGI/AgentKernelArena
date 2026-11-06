"""High-precision routed-experts MoE reference used only for correctness."""
import torch

# Expert weight rows converted to float64 at a time, which bounds the reference's temporary memory.
ROW_CHUNK = 4096


def _matmul_t(x, weight):
    """x @ weight.T in float64."""
    out = torch.empty((x.shape[0], weight.shape[0]), dtype=torch.float64, device=x.device)
    for start in range(0, weight.shape[0], ROW_CHUNK):
        out[:, start:start + ROW_CHUNK] = x @ weight[start:start + ROW_CHUNK].double().t()
    return out


def reference(*, hidden_states, w13, w2, topk_weights, topk_ids):
    num_tokens, hidden = hidden_states.shape
    inter = w13.shape[1] // 2
    out = torch.zeros((num_tokens, hidden), dtype=torch.float64, device=hidden_states.device)
    weights = topk_weights.double()
    for expert in torch.unique(topk_ids).tolist():
        token, slot = (topk_ids == expert).nonzero(as_tuple=True)
        h = _matmul_t(hidden_states[token].double(), w13[expert])
        gate, up = h[:, :inter], h[:, inter:]
        act = gate * torch.sigmoid(gate) * up
        out.index_add_(0, token, _matmul_t(act, w2[expert]) * weights[token, slot].unsqueeze(1))
    return out
