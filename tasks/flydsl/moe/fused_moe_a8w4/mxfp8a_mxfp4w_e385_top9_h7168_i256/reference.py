# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""High-precision reference of the MXFP8-activation x MXFP4-weight routed MoE layer, used only for correctness.

MXFP8 activation quantization (hidden states per token, the stage-1 activation per token and slot), per 1 x 32
block of float32 values: e = biased exponent of max|x| rounded up when its mantissa is at least 1.5 (bit 22),
e8m0 = max(e - 8, 0), code = float8_e4m3fn(x * 2^(127 - e8m0)) rounded to nearest even, value = code *
2^(e8m0 - 127). MXFP4 weights: e2m1 code (low nibble = even element) times 2^(byte - 127) per 32 elements.
Everything else is float64 arithmetic on the dequantized operands, with no intermediate rounding (silu(gate) * up
is quantized from its float64 value rounded once to float32).
"""
import torch

BLOCK = 32
ROW_CHUNK = 2048  # rows of one expert evaluated at a time, to bound float64 temporaries
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def _mxfp8_quant_dequant(x32):
    """MXFP8 quantization of a float32 matrix [rows, cols], dequantized (exactly) in float32."""
    rows, cols = x32.shape
    blocks = x32.view(rows, cols // BLOCK, BLOCK)
    bits = blocks.abs().amax(dim=-1).contiguous().view(torch.int32)
    e8m0 = (((bits + 0x400000) >> 23) & 0xFF) - 8
    e8m0 = e8m0.clamp_min(0)
    up = ((254 - e8m0) << 23).view(torch.float32)      # 2^(127 - e8m0)
    down = torch.exp2(e8m0.to(torch.float32) - 127.0)  # 2^(e8m0 - 127)
    codes = (blocks * up.unsqueeze(-1)).to(torch.float8_e4m3fn)
    return (codes.float() * down.unsqueeze(-1)).view(rows, cols)


def _mxfp4_dequant(codes, scales):
    """[N, K/2] packed e2m1 and [N, K/32] e8m0 -> float64 [N, K]."""
    lut = torch.tensor(_E2M1, dtype=torch.float64, device=codes.device)
    values = torch.stack((lut[(codes & 0xF).long()], lut[(codes >> 4).long()]), dim=-1).reshape(codes.shape[0], -1)
    return values * torch.exp2(scales.to(torch.float64) - 127.0).repeat_interleave(BLOCK, dim=-1)


def reference(*, hidden_states, w1, w1_scale, w2, w2_scale, topk_weights, topk_ids):
    num_tokens, hidden = hidden_states.shape
    num_experts, gate_up, _ = w1.shape
    inter = gate_up // 2
    a1 = torch.cat([_mxfp8_quant_dequant(hidden_states[i:i + ROW_CHUNK].float())
                    for i in range(0, num_tokens, ROW_CHUNK)])
    out = torch.zeros(num_tokens, hidden, dtype=torch.float64, device=hidden_states.device)
    ids = topk_ids.long()
    for expert in range(num_experts):
        tokens, slots = torch.where(ids == expert)
        if tokens.numel() == 0:
            continue
        w_up = _mxfp4_dequant(w1[expert], w1_scale[expert])
        w_down = _mxfp4_dequant(w2[expert], w2_scale[expert])
        for start in range(0, tokens.numel(), ROW_CHUNK):
            token, slot = tokens[start:start + ROW_CHUNK], slots[start:start + ROW_CHUNK]
            h = a1[token].double() @ w_up.T
            gate, up = h[:, :inter], h[:, inter:]
            act = gate * torch.sigmoid(gate) * up
            a2 = _mxfp8_quant_dequant(act.float()).double() * topk_weights[token, slot].double().unsqueeze(-1)
            # A token selects each expert at most once, so the rows of one index_add_ are distinct.
            out.index_add_(0, token, a2 @ w_down.T)
    return out
