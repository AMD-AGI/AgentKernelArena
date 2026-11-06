"""Float64 reference for the MXFP4 routed MoE layer, used only for correctness.

For every (token, slot) routed to expert e:
  x_q     = MXFP4(hidden_states[token])                      activation quantized per 32 values
  gate    = x_q @ W1[e, :intermediate_size]^T,  up = x_q @ W1[e, intermediate_size:]^T
  h       = bf16(silu(gate) * up)                            rounded through float32, as a kernel does
  partial = MXFP4(h) @ W2[e]^T
out[token] = sum over slots of topk_weights[token, slot] * partial.
Every product and sum is taken in float64; MXFP4 is defined in mxfp4_numerics.py.
"""
import torch

from mxfp4_numerics import dequantize, mxfp4_round


def reference(*, hidden_states, w1, w1_scale, w2, w2_scale, topk_weights, topk_ids):
    num_tokens, hidden_size = hidden_states.shape
    intermediate_size = w1.shape[1] // 2
    top_k = topk_ids.shape[1]
    device = hidden_states.device
    x = mxfp4_round(hidden_states)
    expert_of = topk_ids.reshape(-1).long()
    weight_of = topk_weights.reshape(-1).double()
    token_of = torch.arange(num_tokens * top_k, device=device) // top_k
    out = torch.zeros((num_tokens, hidden_size), dtype=torch.float64, device=device)
    for expert in torch.unique(expert_of).tolist():
        pairs = (expert_of == expert).nonzero(as_tuple=True)[0]
        tokens = token_of[pairs]
        gate_up = x[tokens] @ dequantize(w1[expert], w1_scale[expert]).T
        gate, up = gate_up[:, :intermediate_size], gate_up[:, intermediate_size:]
        h = (gate * torch.sigmoid(gate) * up).to(torch.float32).to(torch.bfloat16)
        partial = mxfp4_round(h) @ dequantize(w2[expert], w2_scale[expert]).T
        out.index_add_(0, tokens, partial * weight_of[pairs, None])
    return out
