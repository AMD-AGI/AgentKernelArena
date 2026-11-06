# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""minGPT transformer block (LayerNorm, causal self-attention, LayerNorm, GELU MLP) in float32 Triton.

Seven launches: LayerNorm 1; the qkv projection; causal flash attention reading q, k and v in place
from the qkv rows and writing the heads side by side; the attention output projection plus the
residual; LayerNorm 2; the c_fc projection with minGPT's tanh GELU in its epilogue; and the MLP output
projection plus the residual. Every matrix product uses IEEE float32 inputs with float32 accumulation.
"""
import math

import torch
import triton
import triton.language as tl


@triton.jit
def _layer_norm_kernel(x_ptr, weight_ptr, bias_ptr, y_ptr, stride_x, stride_y, N, eps,
                       BLOCK_N: tl.constexpr):
    """y[row] = (x[row] - mean) / sqrt(biased var + eps) * weight + bias, one program per row."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / N
    centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(centered * centered, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + cols, mask=mask, other=0.0)
    tl.store(y_ptr + row * stride_y + cols, centered * rstd * weight + bias, mask=mask)


@triton.jit
def _linear_kernel(a_ptr, w_ptr, bias_ptr, residual_ptr, c_ptr, M, N, K,
                   stride_am, stride_wn, stride_rm, stride_cm,
                   GELU: tl.constexpr, ADD_RESIDUAL: tl.constexpr,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                   GROUP_M: tl.constexpr):
    """c[M, N] = epilogue(a[M, K] @ w[N, K]^T + bias[N]); K is a multiple of BLOCK_K.

    The epilogue applies minGPT's NewGELU, 0.5 x (1 + tanh(z)) with z = sqrt(2 / pi) (x + 0.044715 x^3),
    written as x / (1 + exp(-2 z)) (the same function), or adds residual[M, N].
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < N
    a_ptrs = a_ptr + offs_m.to(tl.int64)[:, None] * stride_am + offs_k[None, :]
    w_ptrs = w_ptr + offs_n.to(tl.int64)[None, :] * stride_wn + offs_k[:, None]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K // BLOCK_K):
        a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
        w = tl.load(w_ptrs, mask=mask_n[None, :], other=0.0)
        acc = tl.dot(a, w, acc, input_precision="ieee")
        a_ptrs += BLOCK_K
        w_ptrs += BLOCK_K

    acc = acc + tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)[None, :]
    mask = mask_m[:, None] & mask_n[None, :]
    if GELU:
        z = 0.7978845608028654 * (acc + 0.044715 * acc * acc * acc)
        acc = acc / (1.0 + tl.exp(-2.0 * z))
    if ADD_RESIDUAL:
        residual = tl.load(residual_ptr + offs_m.to(tl.int64)[:, None] * stride_rm + offs_n[None, :],
                           mask=mask, other=0.0)
        acc = acc + residual
    tl.store(c_ptr + offs_m.to(tl.int64)[:, None] * stride_cm + offs_n[None, :], acc, mask=mask)


@triton.jit
def _causal_attention_kernel(qkv_ptr, y_ptr, T, stride_qkv, stride_y, qk_scale,
                             N_HEAD: tl.constexpr, N_EMBD: tl.constexpr, HEAD_DIM: tl.constexpr,
                             BLOCK_D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """y[b, t, h*D:(h+1)*D] = softmax over keys j <= t of (q k^T / sqrt(D)) v, per (query block, b, h).

    qk_scale is log2(e) / sqrt(D): the online softmax runs in exp2.
    """
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    batch = pid_bh // N_HEAD
    head = pid_bh % N_HEAD
    row0 = batch.to(tl.int64) * T

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < HEAD_DIM
    mask_m = offs_m < T
    q = tl.load(qkv_ptr + (row0 + offs_m)[:, None] * stride_qkv + head * HEAD_DIM + offs_d[None, :],
                mask=mask_m[:, None] & mask_d[None, :], other=0.0)

    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    # Keys after the block's last query are masked for every row of the block.
    hi = tl.minimum((pid_m + 1) * BLOCK_M, T)
    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < T
        k = tl.load(qkv_ptr + (row0 + offs_n)[None, :] * stride_qkv + N_EMBD + head * HEAD_DIM + offs_d[:, None],
                    mask=mask_n[None, :] & mask_d[:, None], other=0.0)
        s = tl.dot(q, k, input_precision="ieee") * qk_scale
        s = tl.where((offs_n[None, :] <= offs_m[:, None]) & mask_n[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        p = tl.math.exp2(s - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(qkv_ptr + (row0 + offs_n)[:, None] * stride_qkv + 2 * N_EMBD + head * HEAD_DIM + offs_d[None, :],
                    mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p, v, input_precision="ieee")
        m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(y_ptr + (row0 + offs_m)[:, None] * stride_y + head * HEAD_DIM + offs_d[None, :], acc,
             mask=mask_m[:, None] & mask_d[None, :])


_N_HEAD = 8
_LARGE_GEMM = dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, GROUP_M=8, num_warps=8, num_stages=2)
_SMALL_GEMM = dict(BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, GROUP_M=8, num_warps=4, num_stages=2)
_ATTENTION = dict(BLOCK_M=64, BLOCK_N=64, num_warps=4, num_stages=1)


def _linear(a, weight, bias, out, *, gelu=False, residual=None):
    m, k = a.shape
    n = weight.shape[0]
    # 128 x 128 tiles once they fill the GPU about twice over; 64 x 64 tiles for smaller problems.
    config = _LARGE_GEMM if triton.cdiv(m, 128) * triton.cdiv(n, 128) >= 512 else _SMALL_GEMM
    assert k % config["BLOCK_K"] == 0
    grid = (triton.cdiv(m, config["BLOCK_M"]) * triton.cdiv(n, config["BLOCK_N"]),)
    res = out if residual is None else residual
    _linear_kernel[grid](a, weight, bias, res, out, m, n, k, a.stride(0), weight.stride(0), res.stride(0),
                         out.stride(0), GELU=gelu, ADD_RESIDUAL=residual is not None, **config)


def _layer_norm(x, weight, bias, eps, out):
    rows, n = x.shape
    _layer_norm_kernel[(rows,)](x, weight, bias, out, x.stride(0), out.stride(0), n, eps,
                                BLOCK_N=triton.next_power_of_2(n), num_warps=4)


def run(*, x, ln1_weight, ln1_bias, c_attn_weight, c_attn_bias, attn_proj_weight, attn_proj_bias,
        ln2_weight, ln2_bias, c_fc_weight, c_fc_bias, mlp_proj_weight, mlp_proj_bias, ln_eps, out):
    batch, seq_len, n_embd = x.shape
    rows = batch * seq_len
    head_dim = n_embd // _N_HEAD
    x2 = x.view(rows, n_embd)
    normed = torch.empty((rows, n_embd), dtype=x.dtype, device=x.device)
    qkv = torch.empty((rows, 3 * n_embd), dtype=x.dtype, device=x.device)
    heads = torch.empty((rows, n_embd), dtype=x.dtype, device=x.device)
    hidden = torch.empty((rows, n_embd), dtype=x.dtype, device=x.device)
    mlp = torch.empty((rows, c_fc_weight.shape[0]), dtype=x.dtype, device=x.device)

    _layer_norm(x2, ln1_weight, ln1_bias, ln_eps, normed)
    _linear(normed, c_attn_weight, c_attn_bias, qkv)
    grid = (triton.cdiv(seq_len, _ATTENTION["BLOCK_M"]), batch * _N_HEAD)
    _causal_attention_kernel[grid](
        qkv, heads, seq_len, qkv.stride(0), heads.stride(0), 1.4426950408889634 / math.sqrt(head_dim),
        N_HEAD=_N_HEAD, N_EMBD=n_embd, HEAD_DIM=head_dim, BLOCK_D=triton.next_power_of_2(head_dim),
        **_ATTENTION)
    _linear(heads, attn_proj_weight, attn_proj_bias, hidden, residual=x2)
    _layer_norm(hidden, ln2_weight, ln2_bias, ln_eps, normed)
    _linear(normed, c_fc_weight, c_fc_bias, mlp, gelu=True)
    _linear(mlp, mlp_proj_weight, mlp_proj_bias, out.view(rows, n_embd), residual=hidden)
