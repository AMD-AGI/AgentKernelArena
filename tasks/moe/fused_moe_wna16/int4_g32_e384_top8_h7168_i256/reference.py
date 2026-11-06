"""High-precision reference for routed experts with int4 group-quantized weights."""
import torch


def _dequantize(qweight, scale):
    """[n, k // 2] uint8 (low nibble = even k, code offset 8) and [n, k // g] scales -> [n, k] float64."""
    rows, half = qweight.shape
    low = (qweight & 0xF).to(torch.float64)
    high = (qweight >> 4).to(torch.float64)
    codes = torch.stack((low, high), dim=-1).reshape(rows, 2 * half)
    group_size = 2 * half // scale.shape[1]
    return (codes - 8.0) * scale.double().repeat_interleave(group_size, dim=1)


def reference(*, hidden_states, w13_qweight, w13_scale, w2_qweight, w2_scale, topk_weights, topk_ids):
    num_tokens, hidden = hidden_states.shape
    num_experts, gate_up, _ = w13_qweight.shape
    inter = gate_up // 2
    out = torch.zeros((num_tokens, hidden), dtype=torch.float64, device=hidden_states.device)
    for expert in range(num_experts):
        token, slot = (topk_ids == expert).nonzero(as_tuple=True)
        if token.numel() == 0:
            continue
        x = hidden_states[token].double()
        h = x @ _dequantize(w13_qweight[expert], w13_scale[expert]).t()
        gate, up = h[:, :inter], h[:, inter:]
        act = gate * torch.sigmoid(gate) * up
        y = (act @ _dequantize(w2_qweight[expert], w2_scale[expert]).t()) * topk_weights[token, slot].double().unsqueeze(1)
        out.index_add_(0, token, y)
    return out
