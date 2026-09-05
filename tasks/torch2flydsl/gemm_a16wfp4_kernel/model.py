# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Pure-PyTorch reference for the mixed BF16/MXFP4 GEMM ``gemm_a16wfp4``.

Computes ``out = a @ w.T`` where the activation ``a`` (``[M, K]``) arrives in
bf16 and is quantized to MXFP4 *on-the-fly inside the GEMM*, and the weight
``w`` (``[N, K]``) is quantized to MXFP4 (e2m1 values with e8m0 per-1x32 block
scales along K), matching the AMD runtime ``gemm_a16wfp4``:

- activation: bf16 input, MXFP4-quantized per-1x32 inside the kernel
  (``_mxfp4_quant_op`` scale ``floor(log2(amax)) - 2`` with e2m1 round-nearest);
- weight: packed FP4 codes (``[N, K/2]``, two e2m1 values per byte) plus an e8m0
  scale per 32-element K block (``w_scale`` is ``[N, K/32]``, uint8);
- the GEMM accumulates the dequantized fp4 products in fp32 and truncates to
  bf16.

To stay byte-faithful, the reference emulates the kernel's on-the-fly
activation quantization and feeds the kernel the resulting (MXFP4-grid) bf16
activation, so the reference and the hardware kernel operate on identical
operands and differ only by fp32 accumulation order. ``quantize_a16wfp4`` is the
single source of those operands.
"""
import torch
import torch.nn as nn

# MXFP4 block size along K (hardware-fixed) and the largest e2m1 magnitude.
_SCALE_GROUP_SIZE = 32
_FP4_MAX = 6.0
# 0xFF800000 as a signed int32: keeps sign + 8-bit exponent (strips mantissa).
_E8M0_MASK_INT32 = -8388608

# MXFP4 (e2m1) decode table indexed by the 4-bit code (sign in bit 3).
_MXFP4_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def prepare_mxfp4_values(device):
    """Move the immutable decode table once, before a timed GPU callback."""
    global _MXFP4_VALUES
    if _MXFP4_VALUES.device != torch.device(device):
        _MXFP4_VALUES = _MXFP4_VALUES.to(device)


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


def _quantize_mxfp4_blockscale(w):
    """Per-1x32 MXFP4 quantization of ``w`` (``[N, K]``).

    Returns packed FP4 codes (``[N, K/2]``, two e2m1 values per byte) and the
    e8m0 block scales (``[N, K/32]``, uint8)."""
    n, k = w.shape
    groups = k // _SCALE_GROUP_SIZE
    wf = w.float().view(n, groups, _SCALE_GROUP_SIZE)
    amax = wf.abs().amax(dim=-1)
    block_scale = (amax / _FP4_MAX).clamp_min(1e-12)
    e8m0 = (torch.log2(block_scale) + 127.0).round().clamp_(0, 127).to(torch.uint8)
    scale_dec = torch.exp2(e8m0.float() - 127.0).unsqueeze(-1)
    codes = _f32_to_e2m1_codes(wf / scale_dec).view(n, k)
    packed = (codes[:, 1::2].to(torch.int32) << 4) | codes[:, ::2].to(torch.int32)
    packed = packed.to(torch.uint8)
    return packed, e8m0


def _emulate_mxfp4_act_prequant(a):
    """Emulate the kernel's on-the-fly per-1x32 MXFP4 activation quantization
    (``_mxfp4_quant_op``) and return the dequantized bf16 activation.

    The result lies exactly on the MXFP4 grid, so feeding it back to the kernel
    (which re-quantizes it) is idempotent."""
    g = _SCALE_GROUP_SIZE
    x = a.float()
    m, k = x.shape
    ng = k // g
    x2 = x.reshape(m, ng, g)
    amax = x2.abs().amax(dim=-1, keepdim=True)

    amax_i32 = amax.contiguous().view(torch.int32)
    amax_i32 = (amax_i32 + 0x200000) & _E8M0_MASK_INT32
    amax_p2 = amax_i32.view(torch.float32)
    scale_unbiased = torch.clamp(amax_p2.log2().floor() - 2, min=-127, max=127)
    quant_scale = torch.exp2(-scale_unbiased)

    codes = _f32_to_e2m1_codes(x2 * quant_scale)
    table = _MXFP4_VALUES
    values = table[codes.long()]
    a_deq = (values * torch.exp2(scale_unbiased)).reshape(m, k)
    return a_deq.to(torch.bfloat16)


def quantize_a16wfp4(a, w):
    """Quantize a high-precision activation/weight pair to the BF16/MXFP4
    operands the deployed mixed GEMM consumes.

    The activation is pre-rounded onto the MXFP4 grid (emulating the kernel's
    on-the-fly quantization) and returned as bf16; the weight is packed MXFP4.
    Returns ``(x_bf16, w_packed, w_scale)`` with the layouts the AMD runtime
    ``gemm_a16wfp4`` expects (``x_bf16`` is ``[M, K]`` bf16, ``w_packed`` is
    ``[N, K/2]`` uint8, ``w_scale`` is ``[N, K/32]`` e8m0 uint8)."""
    x_bf16 = _emulate_mxfp4_act_prequant(a)
    w_packed, w_scale = _quantize_mxfp4_blockscale(w)
    return x_bf16, w_packed, w_scale


def _dequant_mxfp4(w_packed, w_scale):
    """Decode packed FP4 codes and e8m0 block scales to fp32 weights."""
    n, kh = w_packed.shape
    k = kh * 2
    codes = torch.empty(n, k, dtype=torch.uint8, device=w_packed.device)
    codes[:, ::2] = w_packed & 0xF
    codes[:, 1::2] = w_packed >> 4
    table = _MXFP4_VALUES
    values = table[codes.long()].view(n, k // _SCALE_GROUP_SIZE, _SCALE_GROUP_SIZE)
    scale_dec = torch.exp2(w_scale.float() - 127.0).unsqueeze(-1)
    return (values * scale_dec).view(n, k)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, w):
        x_bf16, w_packed, w_scale = quantize_a16wfp4(a, w)
        x_f32 = x_bf16.float()
        w_f32 = _dequant_mxfp4(w_packed, w_scale)
        out = torch.matmul(x_f32, w_f32.transpose(0, 1))
        return out.to(torch.bfloat16)


def get_inputs():
    # Representative shape (M, N, K) = (128, 2112, 7168); the harness sweeps more
    # shapes (K a multiple of the MXFP4 32-block).
    m, n, k = 128, 2112, 7168
    a = torch.randn(m, k, dtype=torch.bfloat16)
    w = torch.randn(n, k, dtype=torch.bfloat16)
    return [a, w]


def get_init_inputs():
    return []
