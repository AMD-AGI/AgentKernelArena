"""High-precision reference of the fp8 block-scaled routed MoE layer, used only for correctness.

Both activation quantizations follow the operator: per token and 128-wide group, scale = max(max|x|, 1e-10)
/ 448 in float32 and code = fp8_e4m3fn(clamp(x * (1 / scale), -448, 448)) with float32 arithmetic and
round-to-nearest-even. Everything else is float64 arithmetic on the dequantized operands, with no
intermediate rounding (the gate/up projection and silu(gate) * up are not rounded to bfloat16).
"""
import torch

FP8_MAX = 448.0
BLOCK = 128
ROW_CHUNK = 2048  # rows of one expert evaluated at a time, to bound float64 temporaries


def _fp8_group_quant(x32):
    """Per-row 1 x 128 fp8 quantization of a float32 matrix: (fp8 codes, float32 scales [rows, cols / 128])."""
    rows, cols = x32.shape
    groups = x32.view(rows, cols // BLOCK, BLOCK)
    scale = groups.abs().amax(dim=-1).clamp_min(1e-10) / FP8_MAX
    recip = 1.0 / scale
    codes = (groups * recip.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return codes.view(rows, cols), scale


def _dequant_rows(codes, scale):
    """fp8 codes [rows, cols] with float32 scales [rows, cols / 128], in float64."""
    rows, cols = codes.shape
    return (codes.double().view(rows, cols // BLOCK, BLOCK) * scale.double().unsqueeze(-1)).view(rows, cols)


def _dequant_weight(codes, scale):
    """[rows, cols] float8 codes with [rows/128, cols/128] block scales, in float64."""
    full = scale.double().repeat_interleave(BLOCK, dim=0).repeat_interleave(BLOCK, dim=1)
    return codes.double() * full


def reference(*, hidden_states, w1, w1_scale, w2, w2_scale, topk_weights, topk_ids):
    num_tokens, hidden = hidden_states.shape
    num_experts, gate_up, _ = w1.shape
    inter = gate_up // 2
    a1_codes, a1_scale = _fp8_group_quant(hidden_states.float())
    out = torch.zeros(num_tokens, hidden, dtype=torch.float64, device=hidden_states.device)
    ids = topk_ids.long()
    for expert in range(num_experts):
        tokens, slots = torch.where(ids == expert)
        if tokens.numel() == 0:
            continue
        w_up = _dequant_weight(w1[expert], w1_scale[expert])
        w_down = _dequant_weight(w2[expert], w2_scale[expert])
        for start in range(0, tokens.numel(), ROW_CHUNK):
            token, slot = tokens[start:start + ROW_CHUNK], slots[start:start + ROW_CHUNK]
            h = _dequant_rows(a1_codes[token], a1_scale[token]) @ w_up.T
            gate, up = h[:, :inter], h[:, inter:]
            act = gate * torch.sigmoid(gate) * up
            a2_codes, a2_scale = _fp8_group_quant(act.float())
            a2 = _dequant_rows(a2_codes, a2_scale) * topk_weights[token, slot].double().unsqueeze(-1)
            # A token selects each expert at most once, so the rows of one index_add_ are distinct.
            out.index_add_(0, token, a2 @ w_down.T)
    return out
