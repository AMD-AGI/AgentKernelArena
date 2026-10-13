# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Pure-PyTorch reference for the quantized a8w4 fused MoE.

The op is a top-k Mixture-of-Experts feed-forward block with MXFP8
(``float8_e4m3fn``) activations and MXFP4 (``float4_e2m1fn_x2``) expert weights,
both using e8m0 per-1x32 block scales. A softmax router selects ``topk`` experts
per token; stage 1 runs a grouped gate/up GEMM followed by ``silu(gate) * up``;
stage 2 runs the down GEMM and combines the experts with the renormalized router
weights. GEMMs accumulate in fp32 over dequantized operands.

Activations are quantized to MXFP8 (e4m3, e8m0 block scale) and dequantized so
the matmul sees the fp8-rounded values; weights are quantized to MXFP4. The
stage-1 result is re-quantized to MXFP8 before the down GEMM, and the output is
returned in bf16. The MXFP4/MXFP8 rounding and e8m0 block-scale numerics
implemented here match AMD's reference quantizer bit-for-bit.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# MXFP4 (e2m1) decode table indexed by the 4-bit code (sign in bit 3).
_MXFP4_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)
_BLOCK = 32
# log2(F4E2M1_MAX=6) floored -> dtypeMax = 2**2 used as the e8m0 fp4 divisor.
_FP4_DTYPE_MAX = 4.0
# log2(F8E4M3_MAX=448) floored -> dtypeMax = 2**8 used as the e8m0 fp8 divisor.
_FP8_DTYPE_MAX = 256.0


def _f32_to_e8m0(x):
    """Round positive fp32 magnitudes to biased e8m0 exponents (uint8)."""
    u32 = x.contiguous().view(torch.int32)
    exponent = ((u32 >> 23) & 0xFF).view(torch.uint32).to(torch.uint8)
    nan_case = exponent == 0xFF
    round_case = ((u32 & 0x400000) > 0) & (
        ((u32 & 0x200000) > 0) | ((u32 & 0x1FFFFF) > 0) | (exponent > 0)
    )
    exponent[round_case] += 1
    exponent[nan_case] = 0xFF
    return exponent


def _e8m0_to_f32(scale_e8m0_biased):
    """Decode biased e8m0 exponents (uint8) back to fp32 power-of-two scales."""
    scale_e8m0_biased = scale_e8m0_biased.view(torch.uint8)
    zero_case = scale_e8m0_biased == 0
    nan_case = scale_e8m0_biased == 0xFF
    scale_f32 = scale_e8m0_biased.to(torch.int32) << 23
    scale_f32[zero_case] = 0x00400000
    scale_f32[nan_case] = 0x7F800001
    return scale_f32.view(torch.float32)


def _f32_to_e2m1_codes(x):
    """Round fp32 values to MXFP4 (e2m1) 4-bit codes, saturating out-of-range
    magnitudes and handling denormals (adapted from the torchao FP utilities)."""
    EBITS, MBITS = 2, 1
    EBITS_F32, MBITS_F32 = 8, 23
    F32_EXP_BIAS = (1 << (EBITS_F32 - 1)) - 1
    exp_bias = (1 << (EBITS - 1)) - 1
    max_int = (1 << (EBITS + MBITS)) - 1
    sign_mask = 1 << (EBITS + MBITS)
    magic_adder = (1 << (MBITS_F32 - MBITS - 1)) - 1
    max_normal = 2 ** ((1 << EBITS) - 1 - exp_bias) * (
        ((1 << (MBITS + 1)) - 1) / (2**MBITS)
    )
    min_normal = 2 ** (1 - exp_bias)
    denorm_exp = (F32_EXP_BIAS - exp_bias) + (MBITS_F32 - MBITS) + 1
    denorm_mask_int = denorm_exp << MBITS_F32
    denorm_mask_float = torch.tensor(
        denorm_mask_int, dtype=torch.int32
    ).view(torch.float32)

    x = x.float().view(torch.int32)
    sign = x & 0x80000000
    x = x ^ sign
    x = x.view(torch.float)

    saturate_mask = x >= max_normal
    denormal_mask = torch.logical_and(
        torch.logical_not(saturate_mask), x < min_normal
    )
    normal_mask = torch.logical_not(torch.logical_or(saturate_mask, denormal_mask))

    denormal_x = x + denorm_mask_float
    denormal_x = denormal_x.view(torch.int32)
    denormal_x -= denorm_mask_int
    denormal_x = denormal_x.to(torch.uint8)

    normal_x = x.view(torch.int32)
    mant_odd = (normal_x >> (MBITS_F32 - MBITS)) & 1
    val_to_add = ((exp_bias - F32_EXP_BIAS) << MBITS_F32) + magic_adder
    normal_x += val_to_add
    normal_x += mant_odd
    normal_x = normal_x >> (MBITS_F32 - MBITS)
    normal_x = normal_x.to(torch.uint8)

    codes = torch.full_like(x, max_int, dtype=torch.uint8)
    codes = torch.where(denormal_mask, denormal_x, codes)
    codes = torch.where(normal_mask, normal_x, codes)

    sign_lp = sign >> (MBITS_F32 + EBITS_F32 - MBITS - EBITS)
    sign_lp = sign_lp.to(torch.uint8) & sign_mask
    return (codes | sign_lp).to(torch.uint8)


def _mxfp4_dequant(x):
    """MXFP4 per-1x32 e8m0 quantize+dequantize over the last dim, returning the
    fp32 values the hardware GEMM sees."""
    shape = x.shape
    xb = x.float().reshape(-1, _BLOCK)
    max_abs = torch.amax(torch.abs(xb), dim=1)
    scale_e8m0 = _f32_to_e8m0(max_abs / _FP4_DTYPE_MAX)
    scale_f32 = _e8m0_to_f32(scale_e8m0).view(-1, 1)
    codes = _f32_to_e2m1_codes(xb / scale_f32)
    table = _MXFP4_VALUES.to(x.device)
    deq = table[codes.long()] * scale_f32
    return deq.reshape(shape)


def _mxfp8_dequant(x):
    """MXFP8 (e4m3) per-1x32 e8m0 quantize+dequantize over the last dim. The
    kernel reads the fp8-rounded activations and multiplies by the e8m0 block
    scale in the GEMM, so the reference reproduces that rounding here."""
    shape = x.shape
    xb = x.float().reshape(-1, _BLOCK)
    max_abs = torch.amax(torch.abs(xb), dim=1)
    scale_e8m0 = _f32_to_e8m0(max_abs / _FP8_DTYPE_MAX)
    scale_f32 = _e8m0_to_f32(scale_e8m0).view(-1, 1)
    y_fp8 = (xb / scale_f32).to(torch.float8_e4m3fn).float()
    deq = (y_fp8 * scale_f32).reshape(shape)
    return deq.to(torch.bfloat16)


def _grouped_gemm_stage1(acts, weights, topk_ids):
    """Per-expert grouped GEMM: out[b, k] = acts[b] @ weights[topk_ids[b, k]].T."""
    acts = acts.float()
    B, D = acts.shape
    topk = topk_ids.shape[1]
    N = weights.shape[1]
    h = acts.view(B, 1, D).repeat(1, topk, 1)
    out = torch.zeros(B, topk, N, dtype=torch.float32, device=acts.device)
    for e in range(weights.shape[0]):
        mask = topk_ids == e
        if mask.any():
            out[mask] = h[mask] @ weights[e].transpose(0, 1)
    return out


def _grouped_gemm_stage2(acts, weights, topk_ids, topk_weights):
    """Per-expert down GEMM with weighted top-k combine to a single output row."""
    acts = acts.float()
    B, topk = topk_ids.shape
    model_dim = weights.shape[1]
    out = torch.zeros(B, topk, model_dim, dtype=torch.float32, device=acts.device)
    for e in range(weights.shape[0]):
        mask = topk_ids == e
        if mask.any():
            out[mask] = acts[mask] @ weights[e].transpose(0, 1)
    out = out * topk_weights.view(B, topk, 1)
    return out.sum(1)


def route_topk(logits, topk):
    """Softmax router + top-k with renormalized weights. Shared by the harness.

    Ties are broken by ascending expert index via a stable descending sort. The
    bf16 gate produces many duplicate logits across the large expert count, and a
    nondeterministic top-k tie-break would let the reference and the runtime op
    select different experts; the stable order keeps both routings identical.
    """
    gate = torch.softmax(logits.float(), dim=-1)
    order = torch.sort(gate, dim=-1, descending=True, stable=True).indices
    ids = order[..., :topk]
    weights = torch.gather(gate, -1, ids)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights.float(), ids.to(torch.int32)


class Model(nn.Module):
    def __init__(self, model_dim, inter_dim, experts, topk, activation="silu"):
        super().__init__()
        self.model_dim = model_dim
        self.inter_dim = inter_dim
        self.experts = experts
        self.topk = topk
        self.activation = activation
        self.gate = nn.Linear(model_dim, experts, bias=False).to(torch.bfloat16)
        self.w1 = nn.Parameter(
            (torch.randn(experts, 2 * inter_dim, model_dim) / 10).to(torch.bfloat16)
        )
        self.w2 = nn.Parameter(
            (torch.randn(experts, model_dim, inter_dim) / 10).to(torch.bfloat16)
        )

    def forward(self, hidden_states):
        I = self.inter_dim

        logits = self.gate(hidden_states)
        topk_weights, topk_ids = route_topk(logits, self.topk)

        a1 = _mxfp8_dequant(hidden_states)
        w1 = _mxfp4_dequant(self.w1)
        w2 = _mxfp4_dequant(self.w2)

        stage1 = _grouped_gemm_stage1(a1, w1, topk_ids)
        gate, up = stage1.split([I, I], dim=-1)
        stage1 = (F.silu(gate) * up).to(torch.bfloat16)

        a2 = _mxfp8_dequant(stage1.reshape(-1, I)).reshape(
            hidden_states.shape[0], self.topk, I
        )

        out = _grouped_gemm_stage2(a2, w2, topk_ids, topk_weights)
        return out.to(torch.float16).to(torch.bfloat16)


def get_inputs():
    return [torch.randn(16, 7168, dtype=torch.bfloat16)]


def get_init_inputs():
    return [7168, 256, 384, 8]
