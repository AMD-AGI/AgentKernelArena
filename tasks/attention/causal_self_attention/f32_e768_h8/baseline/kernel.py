# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""minGPT CausalSelfAttention forward in float32 Triton.

Three launches: the fused qkv projection (x @ c_attn_weight^T + c_attn_bias), causal
flash attention reading q, k and v in place from the qkv rows and writing the heads side by
side, and the output projection (y @ c_proj_weight^T + c_proj_bias). Every matrix product
uses IEEE float32 inputs with float32 accumulation.
"""
import math

import torch
import triton
import triton.language as tl


@triton.jit
def _linear_bias_kernel(a_ptr, w_ptr, bias_ptr, c_ptr, M, N, K,
                        stride_am, stride_wn, stride_cm,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                        GROUP_M: tl.constexpr):
    """c[M, N] = a[M, K] @ w[N, K]^T + bias[N] (row-major, unit inner strides)."""
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
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = offs_k < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=mask_n[None, :] & k_mask[:, None], other=0.0)
        acc = tl.dot(a, w, acc, input_precision="ieee")
        a_ptrs += BLOCK_K
        w_ptrs += BLOCK_K

    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]
    c_ptrs = c_ptr + offs_m.to(tl.int64)[:, None] * stride_cm + offs_n[None, :]
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _causal_attention_kernel(qkv_ptr, y_ptr, T, stride_qkv, stride_y, qk_scale,
                             N_HEAD: tl.constexpr, N_EMBD: tl.constexpr, HEAD_DIM: tl.constexpr,
                             BLOCK_D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """y[b, t, h*D:(h+1)*D] = softmax_causal(q k^T / sqrt(D)) v for one (query block, batch, head)."""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    batch = pid_bh // N_HEAD
    head = pid_bh % N_HEAD
    row0 = batch.to(tl.int64) * T

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < HEAD_DIM
    mask_m = offs_m < T
    q_ptrs = qkv_ptr + (row0 + offs_m)[:, None] * stride_qkv + head * HEAD_DIM + offs_d[None, :]
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)

    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    # Keys beyond the block's last query row are masked for every row: skip them.
    hi = tl.minimum((pid_m + 1) * BLOCK_M, T)
    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < T
        k_ptrs = qkv_ptr + (row0 + offs_n)[None, :] * stride_qkv + N_EMBD + head * HEAD_DIM + offs_d[:, None]
        k = tl.load(k_ptrs, mask=mask_n[None, :] & mask_d[:, None], other=0.0)
        s = tl.dot(q, k, input_precision="ieee") * qk_scale
        s = tl.where(offs_n[None, :] <= offs_m[:, None], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        p = tl.math.exp2(s - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v_ptrs = qkv_ptr + (row0 + offs_n)[:, None] * stride_qkv + 2 * N_EMBD + head * HEAD_DIM + offs_d[None, :]
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p, v, input_precision="ieee")
        m_i = m_new

    acc = acc / l_i[:, None]
    y_ptrs = y_ptr + (row0 + offs_m)[:, None] * stride_y + head * HEAD_DIM + offs_d[None, :]
    tl.store(y_ptrs, acc, mask=mask_m[:, None] & mask_d[None, :])


_GEMM = dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, GROUP_M=8, num_warps=8, num_stages=2)
_ATTENTION = dict(BLOCK_M=64, BLOCK_N=64, num_warps=4, num_stages=1)
_N_HEAD = 8


def _linear(a, weight, bias, out):
    m, k = a.shape
    n = weight.shape[0]
    grid = (triton.cdiv(m, _GEMM["BLOCK_M"]) * triton.cdiv(n, _GEMM["BLOCK_N"]),)
    _linear_bias_kernel[grid](a, weight, bias, out, m, n, k, a.stride(0), weight.stride(0), out.stride(0), **_GEMM)


def run(*, x, c_attn_weight, c_attn_bias, c_proj_weight, c_proj_bias, out):
    batch, seq_len, n_embd = x.shape
    rows = batch * seq_len
    head_dim = n_embd // _N_HEAD
    qkv = torch.empty((rows, 3 * n_embd), dtype=x.dtype, device=x.device)
    y = torch.empty((rows, n_embd), dtype=x.dtype, device=x.device)

    _linear(x.view(rows, n_embd), c_attn_weight, c_attn_bias, qkv)
    grid = (triton.cdiv(seq_len, _ATTENTION["BLOCK_M"]), batch * _N_HEAD)
    _causal_attention_kernel[grid](
        qkv, y, seq_len, qkv.stride(0), y.stride(0), 1.4426950408889634 / math.sqrt(head_dim),
        N_HEAD=_N_HEAD, N_EMBD=n_embd, HEAD_DIM=head_dim, BLOCK_D=triton.next_power_of_2(head_dim),
        **_ATTENTION)
    _linear(y, c_proj_weight, c_proj_bias, out.view(rows, n_embd))
