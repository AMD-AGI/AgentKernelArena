# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Vision attention block (nn.MultiheadAttention over image positions + residual LayerNorm)
in float32 Triton.

Three launches per call:
  1. the fused q/k/v input projection, reading the channel-major [batch, embed_dim, hw] image
     directly and writing token-major qkv rows [batch, hw, 3 * embed_dim];
  2. non-causal flash attention per (query block, batch, head), heads written side by side;
  3. the output projection with its bias, the residual add and the LayerNorm over the 128
     channels in the epilogue, stored back channel-major into [batch, embed_dim, hw].
Every matrix product uses IEEE float32 inputs with float32 accumulation.
"""
import math

import torch
import triton
import triton.language as tl


@triton.jit
def _in_proj_kernel(x_ptr, w_ptr, bias_ptr, qkv_ptr, HW, stride_xb, stride_qkvb,
                    EMBED: tl.constexpr, N_OUT: tl.constexpr,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """qkv[b, l, n] = sum_c x[b, c, l] * w[n, c] + bias[n] for one (position block, n block, batch)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2).to(tl.int64)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < HW
    x_base = x_ptr + pid_b * stride_xb

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, EMBED, BLOCK_K):
        # x is channel-major: the [positions, channels] tile is contiguous along positions.
        a = tl.load(x_base + (k0 + offs_k)[None, :] * HW + offs_m[:, None], mask=mask_m[:, None], other=0.0)
        w = tl.load(w_ptr + offs_n[None, :] * EMBED + (k0 + offs_k)[:, None])
        acc = tl.dot(a, w, acc, input_precision="ieee")

    acc = acc + tl.load(bias_ptr + offs_n)[None, :]
    out_ptrs = qkv_ptr + pid_b * stride_qkvb + offs_m[:, None] * N_OUT + offs_n[None, :]
    tl.store(out_ptrs, acc, mask=mask_m[:, None])


@triton.jit
def _attention_kernel(qkv_ptr, o_ptr, HW, stride_qkvb, stride_ob, qk_scale,
                      EMBED: tl.constexpr, NUM_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """o[b, l, h*D:(h+1)*D] = softmax(q k^T / sqrt(D)) v over all hw keys of image b, head h."""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    batch = (pid_bh // NUM_HEADS).to(tl.int64)
    head = pid_bh % NUM_HEADS
    row_stride: tl.constexpr = 3 * EMBED

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_m = offs_m < HW
    base = qkv_ptr + batch * stride_qkvb + head * HEAD_DIM
    q = tl.load(base + offs_m[:, None] * row_stride + offs_d[None, :], mask=mask_m[:, None], other=0.0)

    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    for start_n in range(0, HW, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < HW
        k = tl.load(base + EMBED + offs_n[None, :] * row_stride + offs_d[:, None], mask=mask_n[None, :], other=0.0)
        s = tl.dot(q, k, input_precision="ieee") * qk_scale
        s = tl.where(mask_n[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        p = tl.math.exp2(s - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(base + 2 * EMBED + offs_n[:, None] * row_stride + offs_d[None, :], mask=mask_n[:, None],
                    other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p, v, input_precision="ieee")
        m_i = m_new

    acc = acc / l_i[:, None]
    o_ptrs = o_ptr + batch * stride_ob + offs_m[:, None] * EMBED + head * HEAD_DIM + offs_d[None, :]
    tl.store(o_ptrs, acc, mask=mask_m[:, None])


@triton.jit
def _out_proj_norm_kernel(attn_ptr, w_ptr, bias_ptr, x_ptr, norm_w_ptr, norm_b_ptr, out_ptr, HW,
                          stride_ab, stride_xb, eps,
                          EMBED: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    """out[b, :, l] = LayerNorm(attn[b, l] @ w^T + bias + x[b, :, l]) for one (position block, batch)."""
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1).to(tl.int64)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = tl.arange(0, EMBED)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < HW
    a_base = attn_ptr + pid_b * stride_ab

    acc = tl.zeros((BLOCK_M, EMBED), dtype=tl.float32)
    for k0 in range(0, EMBED, BLOCK_K):
        a = tl.load(a_base + offs_m[:, None] * EMBED + (k0 + offs_k)[None, :], mask=mask_m[:, None], other=0.0)
        w = tl.load(w_ptr + offs_c[None, :] * EMBED + (k0 + offs_k)[:, None])
        acc = tl.dot(a, w, acc, input_precision="ieee")

    image = pid_b * stride_xb + offs_c[None, :] * HW + offs_m[:, None]  # channel-major [EMBED, HW] tile
    residual = tl.load(x_ptr + image, mask=mask_m[:, None], other=0.0)
    z = acc + tl.load(bias_ptr + offs_c)[None, :] + residual
    mean = tl.sum(z, axis=1) / EMBED
    centered = z - mean[:, None]
    variance = tl.sum(centered * centered, axis=1) / EMBED
    rstd = 1.0 / tl.sqrt(variance + eps)
    y = centered * rstd[:, None] * tl.load(norm_w_ptr + offs_c)[None, :] + tl.load(norm_b_ptr + offs_c)[None, :]
    tl.store(out_ptr + image, y, mask=mask_m[:, None])


_NUM_HEADS = 4
_IN_PROJ = dict(BLOCK_M=64, BLOCK_N=128, BLOCK_K=32, num_warps=4, num_stages=2)
_ATTENTION = dict(BLOCK_M=64, BLOCK_N=64, num_warps=4, num_stages=1)
_OUT_PROJ = dict(BLOCK_M=32, BLOCK_K=32, num_warps=4, num_stages=2)


def run(*, x, in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias, norm_weight, norm_bias, epsilon,
        out):
    batch, embed_dim, hw = x.shape
    head_dim = embed_dim // _NUM_HEADS
    qkv = torch.empty((batch, hw, 3 * embed_dim), dtype=x.dtype, device=x.device)
    attention = torch.empty((batch, hw, embed_dim), dtype=x.dtype, device=x.device)

    grid = (triton.cdiv(hw, _IN_PROJ["BLOCK_M"]), 3 * embed_dim // _IN_PROJ["BLOCK_N"], batch)
    _in_proj_kernel[grid](x, in_proj_weight, in_proj_bias, qkv, hw, x.stride(0), qkv.stride(0),
                          EMBED=embed_dim, N_OUT=3 * embed_dim, **_IN_PROJ)

    grid = (triton.cdiv(hw, _ATTENTION["BLOCK_M"]), batch * _NUM_HEADS)
    _attention_kernel[grid](qkv, attention, hw, qkv.stride(0), attention.stride(0),
                            1.4426950408889634 / math.sqrt(head_dim),
                            EMBED=embed_dim, NUM_HEADS=_NUM_HEADS, HEAD_DIM=head_dim, **_ATTENTION)

    grid = (triton.cdiv(hw, _OUT_PROJ["BLOCK_M"]), batch)
    _out_proj_norm_kernel[grid](attention, out_proj_weight, out_proj_bias, x, norm_weight, norm_bias, out, hw,
                                attention.stride(0), x.stride(0), epsilon, EMBED=embed_dim, **_OUT_PROJ)
