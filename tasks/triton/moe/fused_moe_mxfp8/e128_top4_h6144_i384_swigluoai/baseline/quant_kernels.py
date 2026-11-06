# SPDX-License-Identifier: Apache-2.0
"""MXFP8 activation quantization and SwiGLU-OAI Triton kernels of SGLang's MiniMax-M3 MXFP8 MoE path.

Vendored from sgl-project/sglang (python/sglang/kernels/ops/quantization/mxfp8_amd_gfx95.py and
python/sglang/kernels/ops/moe/minimax_m3_swiglu.py): the per-1x32-block E8M0 quantization kernel and the
split-layout SwiGLU-OAI kernel, with their launchers stripped of dispatch to PyTorch fallbacks.
"""
import torch
import triton
import triton.language as tl

MXFP8_BLOCK_SIZE = 32


@triton.jit
def _mxfp8_quant_kernel(
    x_ptr,
    xq_ptr,
    s_ptr,
    M,
    K,
    sxm,
    sxk,
    sqm,
    sqk,
    ssm,
    ssk,
    BLOCK_M: tl.constexpr,
):
    """Per-32-block E8M0 scale + FP8-E4M3 quant, one program per ``[BLOCK_M, 32]``."""
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)  # which 32-element block along K
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_b * 32 + tl.arange(0, 32)
    m_mask = offs_m < M
    x = tl.load(
        x_ptr + offs_m[:, None] * sxm + offs_k[None, :] * sxk,
        mask=m_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-30)  # [BLOCK_M]
    # Round the E8M0 exponent up (ceil(log2(amax / e4m3_max))) so the block amax
    # stays inside the e4m3 range and the full dynamic range is used.
    sb = tl.ceil(tl.log2(amax / 448.0)) + 127.0
    sb = tl.minimum(tl.maximum(sb, 0.0), 254.0)
    descale = tl.exp2(sb - 127.0)
    xq = tl.clamp(x / descale[:, None], -448.0, 448.0).to(xq_ptr.dtype.element_ty)
    tl.store(
        xq_ptr + offs_m[:, None] * sqm + offs_k[None, :] * sqk,
        xq,
        mask=m_mask[:, None],
    )
    tl.store(s_ptr + offs_m * ssm + pid_b * ssk, sb.to(tl.uint8), mask=m_mask)


def mxfp8_e4m3_quantize(x):
    """Fused 2D MXFP8 quant (row-major [M, K//32] UE8M0 scales) of a contiguous tensor."""
    M, K = x.shape
    xq = torch.empty((M, K), dtype=torch.float8_e4m3fn, device=x.device)
    scales = torch.empty((M, K // MXFP8_BLOCK_SIZE), dtype=torch.uint8, device=x.device)
    BLOCK_M = 64
    grid = (triton.cdiv(M, BLOCK_M), K // MXFP8_BLOCK_SIZE)
    _mxfp8_quant_kernel[grid](
        x,
        xq,
        scales,
        M,
        K,
        x.stride(0),
        x.stride(1),
        xq.stride(0),
        xq.stride(1),
        scales.stride(0),
        scales.stride(1),
        BLOCK_M=BLOCK_M,
    )
    return xq, scales


@triton.jit
def _swiglu_oai_kernel(
    g_ptr,
    out_ptr,
    n_inter,
    stride_gm,
    stride_gn,
    stride_om,
    stride_on,
    alpha,
    beta,
    limit,
    HAS_LIMIT: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    row = tl.program_id(0)
    pid_i = tl.program_id(1)
    cols = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    mask = cols < n_inter
    gate = tl.load(g_ptr + row * stride_gm + cols * stride_gn, mask=mask, other=0.0).to(
        tl.float32
    )
    up = tl.load(
        g_ptr + row * stride_gm + (n_inter + cols) * stride_gn,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    if HAS_LIMIT:
        gate = tl.minimum(gate, limit)
        up = tl.minimum(tl.maximum(up, -limit), limit)
    out = gate * tl.sigmoid(alpha * gate) * (up + beta)
    tl.store(
        out_ptr + row * stride_om + cols * stride_on,
        out.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def swiglu_oai_split(gate_up, alpha, beta, limit):
    """SwiGLU-OAI on a split-layout ``[M, 2I]`` tensor -> ``[M, I]`` (fp32 math, same dtype out)."""
    two_i = gate_up.shape[-1]
    n_inter = two_i // 2
    m = gate_up.shape[0]
    out = torch.empty((m, n_inter), dtype=gate_up.dtype, device=gate_up.device)
    # Upstream's adaptive tile: 512 wide once the per-rank slice is large, else 256.
    block_i = 512 if n_inter >= 2048 else 256
    grid = (m, triton.cdiv(n_inter, block_i))
    _swiglu_oai_kernel[grid](
        gate_up,
        out,
        n_inter,
        gate_up.stride(0),
        gate_up.stride(1),
        out.stride(0),
        out.stride(1),
        float(alpha),
        float(beta),
        float(limit),
        HAS_LIMIT=True,
        BLOCK_I=block_i,
        num_warps=4,
    )
    return out
