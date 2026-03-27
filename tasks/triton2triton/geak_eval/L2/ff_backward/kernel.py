#!/usr/bin/env python3
"""
Fused Gated MLP (Feed-Forward) Backward Kernel — Pure Triton

Implements the full backward pass for gated feed-forward networks (SwiGLU)
entirely in Triton, including all GEMMs.  Derived from the GEAK OE profiler
reference (geak_oe_profiler_pure_triton.py).

Kernels:
  _fused_dg_gating_kernel : dg = dy @ w_down.T, then dh0/dh1 with activation grad
  _fused_dx_kernel        : dx = dh0 @ w_gate + dh1 @ w_value
  _fused_dw_up_kernel     : dw_up = [dh0.T @ x ; dh1.T @ x]
  _dw_down_kernel         : dw_down = g.T @ dy
"""

import math

import torch
import triton
import triton.language as tl


# ============================================================================
# REFERENCE HELPERS (PyTorch, for correctness checking)
# ============================================================================


def silu_backward(x, grad):
    sigmoid_x = torch.sigmoid(x)
    return grad * (sigmoid_x + x * sigmoid_x * (1 - sigmoid_x))


def _pytorch_backward_reference(dy, x, w_up, w_down, h0, h1, a, g, activation='silu'):
    N_half = h0.shape[1]

    dg = torch.matmul(dy, w_down.t())

    if activation == 'silu':
        da = silu_backward(h0, dg * h1)
    else:
        da = dg * h1

    dh0 = da
    dh1 = dg * a

    w_gate = w_up[:N_half, :]
    w_value = w_up[N_half:, :]
    dx = torch.matmul(dh0, w_gate) + torch.matmul(dh1, w_value)

    dw_gate = torch.matmul(dh0.t(), x)
    dw_value = torch.matmul(dh1.t(), x)
    dw_up = torch.cat([dw_gate, dw_value], dim=0)

    dw_down = torch.matmul(g.t(), dy)

    return dx, dw_up, dw_down


def ff_fused_gated_forward(x, w_up, w_down, activation='silu'):
    N = w_up.shape[0]
    N_half = N // 2

    w_gate = w_up[:N_half, :]
    w_value = w_up[N_half:, :]

    h0 = torch.matmul(x, w_gate.t())
    h1 = torch.matmul(x, w_value.t())

    if activation == 'silu':
        a = torch.nn.functional.silu(h0)
    elif activation == 'gelu':
        a = torch.nn.functional.gelu(h0)
    else:
        a = h0

    g = a * h1
    y = g @ w_down

    return y, h0, h1, a, g


# ============================================================================
# TRITON KERNELS — verbatim from geak_oe_profiler_pure_triton.py
# ============================================================================


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 4, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 4, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 4, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 4, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_stages=1, num_warps=8),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_stages=1, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=2, num_warps=8),
    ],
    key=['M', 'N_half', 'K'],
)
@triton.jit
def _fused_dg_gating_kernel(
    dy_ptr, w_down_ptr,
    h0_ptr, h1_ptr, a_ptr,
    dh0_ptr, dh1_ptr,
    M, N_half, K,
    stride_dym, stride_dyk,
    stride_wk, stride_wn,
    stride_h0m, stride_h0n,
    stride_h1m, stride_h1n,
    stride_am, stride_an,
    stride_dh0m, stride_dh0n,
    stride_dh1m, stride_dh1n,
    USE_SILU: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Fused kernel: dg = dy @ w_down.T, then compute dh0 and dh1"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_offs = k + offs_k
        k_mask = k_offs < K

        dy_ptrs = dy_ptr + offs_m[:, None] * stride_dym + k_offs[None, :] * stride_dyk
        dy_mask = (offs_m[:, None] < M) & k_mask[None, :]
        dy_block = tl.load(dy_ptrs, mask=dy_mask, other=0.0)

        w_ptrs = w_down_ptr + offs_n[None, :] * stride_wk + k_offs[:, None] * stride_wn
        w_mask = k_mask[:, None] & (offs_n[None, :] < N_half)
        w_block = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc += tl.dot(dy_block, w_block)

    dg = acc

    h0_ptrs = h0_ptr + offs_m[:, None] * stride_h0m + offs_n[None, :] * stride_h0n
    h1_ptrs = h1_ptr + offs_m[:, None] * stride_h1m + offs_n[None, :] * stride_h1n
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N_half)
    h0 = tl.load(h0_ptrs, mask=mask, other=0.0).to(tl.float32)
    h1 = tl.load(h1_ptrs, mask=mask, other=0.0).to(tl.float32)
    a = tl.load(a_ptrs, mask=mask, other=0.0).to(tl.float32)

    if USE_SILU:
        sigmoid_h0 = 1.0 / (1.0 + tl.exp(-h0))
        silu_grad = sigmoid_h0 + h0 * sigmoid_h0 * (1.0 - sigmoid_h0)
        dh0 = dg * h1 * silu_grad
    else:
        dh0 = dg * h1

    dh1 = dg * a

    dh0_ptrs = dh0_ptr + offs_m[:, None] * stride_dh0m + offs_n[None, :] * stride_dh0n
    dh1_ptrs = dh1_ptr + offs_m[:, None] * stride_dh1m + offs_n[None, :] * stride_dh1n

    tl.store(dh0_ptrs, dh0.to(dh0_ptr.dtype.element_ty), mask=mask)
    tl.store(dh1_ptrs, dh1.to(dh1_ptr.dtype.element_ty), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 4, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 4, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 4, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_stages=1, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=2, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _fused_dx_kernel(
    dh0_ptr, dh1_ptr, w_gate_ptr, w_value_ptr, dx_ptr,
    M, N, K,
    stride_dh0m, stride_dh0n,
    stride_dh1m, stride_dh1n,
    stride_wgn, stride_wgk,
    stride_wvn, stride_wvk,
    stride_dxm, stride_dxk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Fused kernel: dx = dh0 @ w_gate + dh1 @ w_value"""
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    for n in range(0, N, BLOCK_N):
        n_offs = n + offs_n
        n_mask = n_offs < N

        dh0_ptrs = dh0_ptr + offs_m[:, None] * stride_dh0m + n_offs[None, :] * stride_dh0n
        dh1_ptrs = dh1_ptr + offs_m[:, None] * stride_dh1m + n_offs[None, :] * stride_dh1n

        mask_mn = (offs_m[:, None] < M) & n_mask[None, :]
        dh0_block = tl.load(dh0_ptrs, mask=mask_mn, other=0.0)
        dh1_block = tl.load(dh1_ptrs, mask=mask_mn, other=0.0)

        wg_ptrs = w_gate_ptr + n_offs[:, None] * stride_wgn + offs_k[None, :] * stride_wgk
        wv_ptrs = w_value_ptr + n_offs[:, None] * stride_wvn + offs_k[None, :] * stride_wvk

        mask_nk = n_mask[:, None] & (offs_k[None, :] < K)
        wg_block = tl.load(wg_ptrs, mask=mask_nk, other=0.0)
        wv_block = tl.load(wv_ptrs, mask=mask_nk, other=0.0)

        acc += tl.dot(dh0_block, wg_block)
        acc += tl.dot(dh1_block, wv_block)

    dx_ptrs = dx_ptr + offs_m[:, None] * stride_dxm + offs_k[None, :] * stride_dxk
    mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    tl.store(dx_ptrs, acc.to(dx_ptr.dtype.element_ty), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 4, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 4, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 4, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_stages=1, num_warps=8),
        triton.Config({'BLOCK_M': 4, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=1, num_warps=8),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=1, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=2, num_warps=8),
    ],
    key=['M', 'N_half', 'K'],
)
@triton.jit
def _fused_dw_up_kernel(
    dh0_ptr, dh1_ptr, x_ptr, dw_up_ptr,
    M, N_half, K,
    stride_dh0m, stride_dh0n,
    stride_dh1m, stride_dh1n,
    stride_xm, stride_xk,
    stride_dwn, stride_dwk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Fused kernel: dw_gate = dh0.T @ x, dw_value = dh1.T @ x, then concat"""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    is_value = tl.program_id(2)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_m = tl.arange(0, BLOCK_M)

    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)

    if is_value == 0:
        dh_ptr = dh0_ptr
        stride_dhm = stride_dh0m
        stride_dhn = stride_dh0n
    else:
        dh_ptr = dh1_ptr
        stride_dhm = stride_dh1m
        stride_dhn = stride_dh1n

    for m in range(0, M, BLOCK_M):
        m_offs = m + offs_m
        m_mask = m_offs < M

        dh_ptrs = dh_ptr + m_offs[:, None] * stride_dhm + offs_n[None, :] * stride_dhn
        mask_mn = m_mask[:, None] & (offs_n[None, :] < N_half)
        dh_block = tl.load(dh_ptrs, mask=mask_mn, other=0.0)

        x_ptrs = x_ptr + m_offs[:, None] * stride_xm + offs_k[None, :] * stride_xk
        mask_mk = m_mask[:, None] & (offs_k[None, :] < K)
        x_block = tl.load(x_ptrs, mask=mask_mk, other=0.0)

        acc += tl.dot(tl.trans(dh_block), x_block)

    out_n_offs = offs_n + is_value * N_half
    dw_ptrs = dw_up_ptr + out_n_offs[:, None] * stride_dwn + offs_k[None, :] * stride_dwk
    mask = (offs_n[:, None] < N_half) & (offs_k[None, :] < K)
    tl.store(dw_ptrs, acc.to(dw_up_ptr.dtype.element_ty), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 4, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 4, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 4, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_stages=1, num_warps=8),
        triton.Config({'BLOCK_M': 4, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=1, num_warps=8),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=1, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=2, num_warps=8),
    ],
    key=['M', 'N_half', 'K'],
)
@triton.jit
def _dw_down_kernel(
    g_ptr, dy_ptr, dw_down_ptr,
    M, N_half, K,
    stride_gm, stride_gn,
    stride_dym, stride_dyk,
    stride_dwn, stride_dwk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Compute dw_down = g.T @ dy"""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_m = tl.arange(0, BLOCK_M)

    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)

    for m in range(0, M, BLOCK_M):
        m_offs = m + offs_m
        m_mask = m_offs < M

        g_ptrs = g_ptr + m_offs[:, None] * stride_gm + offs_n[None, :] * stride_gn
        mask_mn = m_mask[:, None] & (offs_n[None, :] < N_half)
        g_block = tl.load(g_ptrs, mask=mask_mn, other=0.0)

        dy_ptrs = dy_ptr + m_offs[:, None] * stride_dym + offs_k[None, :] * stride_dyk
        mask_mk = m_mask[:, None] & (offs_k[None, :] < K)
        dy_block = tl.load(dy_ptrs, mask=mask_mk, other=0.0)

        acc += tl.dot(tl.trans(g_block), dy_block)

    dw_ptrs = dw_down_ptr + offs_n[:, None] * stride_dwn + offs_k[None, :] * stride_dwk
    mask = (offs_n[:, None] < N_half) & (offs_k[None, :] < K)
    tl.store(dw_ptrs, acc.to(dw_down_ptr.dtype.element_ty), mask=mask)


# ============================================================================
# PYTHON WRAPPER
# ============================================================================


def ff_fused_gated_backward_triton(
    dy, x, w_up, w_down, h0, h1, a, g, activation='silu',
):
    """Full backward pass for fused gated feed-forward, all in Triton."""
    M, K = x.shape
    N = w_up.shape[0]
    N_half = N // 2
    K_out = dy.shape[1]

    if not dy.is_contiguous(): dy = dy.contiguous()
    if not x.is_contiguous(): x = x.contiguous()
    if not w_up.is_contiguous(): w_up = w_up.contiguous()
    if not w_down.is_contiguous(): w_down = w_down.contiguous()
    if not h0.is_contiguous(): h0 = h0.contiguous()
    if not h1.is_contiguous(): h1 = h1.contiguous()
    if not a.is_contiguous(): a = a.contiguous()
    if not g.is_contiguous(): g = g.contiguous()

    dh0 = torch.empty((M, N_half), dtype=x.dtype, device=x.device)
    dh1 = torch.empty((M, N_half), dtype=x.dtype, device=x.device)
    dx = torch.empty_like(x)
    dw_up = torch.empty_like(w_up)
    dw_down = torch.empty((N_half, K_out), dtype=w_down.dtype, device=w_down.device)

    USE_SILU = activation == 'silu'

    def grid_dg(META):
        return (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N_half, META['BLOCK_N']))

    _fused_dg_gating_kernel[grid_dg](
        dy, w_down,
        h0, h1, a,
        dh0, dh1,
        M, N_half, K_out,
        dy.stride(0), dy.stride(1),
        w_down.stride(0), w_down.stride(1),
        h0.stride(0), h0.stride(1),
        h1.stride(0), h1.stride(1),
        a.stride(0), a.stride(1),
        dh0.stride(0), dh0.stride(1),
        dh1.stride(0), dh1.stride(1),
        USE_SILU=USE_SILU,
    )

    w_gate = w_up[:N_half, :]
    w_value = w_up[N_half:, :]

    def grid_dx(META):
        return (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(K, META['BLOCK_K']))

    _fused_dx_kernel[grid_dx](
        dh0, dh1, w_gate, w_value, dx,
        M, N_half, K,
        dh0.stride(0), dh0.stride(1),
        dh1.stride(0), dh1.stride(1),
        w_gate.stride(0), w_gate.stride(1),
        w_value.stride(0), w_value.stride(1),
        dx.stride(0), dx.stride(1),
    )

    def grid_dw_up(META):
        return (triton.cdiv(N_half, META['BLOCK_N']), triton.cdiv(K, META['BLOCK_K']), 2)

    _fused_dw_up_kernel[grid_dw_up](
        dh0, dh1, x, dw_up,
        M, N_half, K,
        dh0.stride(0), dh0.stride(1),
        dh1.stride(0), dh1.stride(1),
        x.stride(0), x.stride(1),
        dw_up.stride(0), dw_up.stride(1),
    )

    def grid_dw_down(META):
        return (triton.cdiv(N_half, META['BLOCK_N']), triton.cdiv(K_out, META['BLOCK_K']))

    _dw_down_kernel[grid_dw_down](
        g, dy, dw_down,
        M, N_half, K_out,
        g.stride(0), g.stride(1),
        dy.stride(0), dy.stride(1),
        dw_down.stride(0), dw_down.stride(1),
    )

    return dx, dw_up, dw_down


# ============================================================================
# ENTRY POINTS (triton_op / torch_op for GEAK harness)
# ============================================================================


def triton_op(M, N, K, x, w_up, w_down, dy, activation='silu'):
    """Run forward then Triton backward, return (dx, dw_up, dw_down)."""
    y, h0, h1, a, g = ff_fused_gated_forward(x, w_up, w_down, activation)
    return ff_fused_gated_backward_triton(dy, x, w_up, w_down, h0, h1, a, g, activation)


def torch_op(M, N, K, x, w_up, w_down, dy, activation='silu'):
    """Run forward then PyTorch reference backward."""
    y, h0, h1, a, g = ff_fused_gated_forward(x, w_up, w_down, activation)
    return _pytorch_backward_reference(dy, x, w_up, w_down, h0, h1, a, g, activation)


# ============================================================================
# TEST CONFIGURATIONS
# ============================================================================

# Configs from geak_oe_profiler_pure_triton.py correctness tests: (M, N, K)
# N = 2*N_half (gate + value concatenated in w_up)
EVAL_CONFIGS = [
    (4, 64, 32),
    (8, 128, 64),
    (16, 256, 128),
    (32, 512, 256),
    (4, 4096, 2048),
    (16, 4096, 2048),
    (4, 16384, 3072),
]

PROFILE_CONFIGS = [
    (16, 256, 128),
    (4, 4096, 2048),
    (4, 16384, 3072),
]

DTYPE = torch.float32
ACTIVATION = "silu"


# ============================================================================
# TEST HARNESS
# ============================================================================


def get_inputs(M, K, N, dtype=DTYPE, device="cuda"):
    """Generate inputs for the backward kernel. N = 2*N_half."""
    N_half = N // 2
    x = torch.randn(M, K, device=device, dtype=dtype)
    w_up = torch.randn(N, K, device=device, dtype=dtype)
    w_down = torch.randn(N_half, K, device=device, dtype=dtype)
    dy = torch.randn(M, K, device=device, dtype=dtype)
    return x, w_up, w_down, dy


def check_correctness(M, K, N, activation=ACTIVATION, dtype=DTYPE) -> dict:
    try:
        x, w_up, w_down, dy = get_inputs(M, K, N, dtype)

        dx_tri, dwup_tri, dwdown_tri = triton_op(M, N, K, x, w_up, w_down, dy, activation)
        dx_ref, dwup_ref, dwdown_ref = torch_op(M, N, K, x, w_up, w_down, dy, activation)

        def rel_diff(a, b):
            max_diff = (a - b).abs().max().item()
            max_val = max(a.abs().max().item(), b.abs().max().item())
            return max_diff / max_val if max_val > 0 else max_diff

        rd_dx = rel_diff(dx_tri, dx_ref)
        rd_dwup = rel_diff(dwup_tri, dwup_ref)
        rd_dwdown = rel_diff(dwdown_tri, dwdown_ref)

        correct = rd_dx < 0.01 and rd_dwup < 0.01 and rd_dwdown < 0.01
        return {
            "correct": correct,
            "rel_dx": rd_dx, "rel_dwup": rd_dwup, "rel_dwdown": rd_dwdown,
            "error": None,
        }
    except Exception as e:
        import traceback
        return {"correct": False, "error": str(e) + "\n" + traceback.format_exc()}


def benchmark_config(M, K, N, activation=ACTIVATION, warmup=50, iters=200) -> dict:
    import time
    x, w_up, w_down, dy = get_inputs(M, K, N)

    y, h0, h1, a, g = ff_fused_gated_forward(x, w_up, w_down, activation)

    # Torch reference
    for _ in range(warmup):
        _pytorch_backward_reference(dy, x, w_up, w_down, h0, h1, a, g, activation)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        _pytorch_backward_reference(dy, x, w_up, w_down, h0, h1, a, g, activation)
    torch.cuda.synchronize()
    torch_ms = (time.perf_counter() - start) * 1000 / iters

    # Triton
    for _ in range(warmup):
        ff_fused_gated_backward_triton(dy, x, w_up, w_down, h0, h1, a, g, activation)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        ff_fused_gated_backward_triton(dy, x, w_up, w_down, h0, h1, a, g, activation)
    torch.cuda.synchronize()
    triton_ms = (time.perf_counter() - start) * 1000 / iters

    return {
        "torch_ms": torch_ms,
        "triton_ms": triton_ms,
        "speedup": torch_ms / triton_ms if triton_ms > 0 else 0.0,
    }


def evaluate(configs=None, warmup=50, iters=200, verbose=True) -> dict:
    configs = configs or EVAL_CONFIGS
    results, failures = [], []

    if verbose:
        print(f"{'Config (M,N,K)':<22} {'Correct':>8} {'Torch':>10} {'Triton':>10} {'Speedup':>10}")
        print("-" * 62)

    for M, N, K in configs:
        corr = check_correctness(M, K, N)
        if not corr["correct"]:
            failures.append({"config": (M, N, K), **corr})
            if verbose:
                err = corr["error"] or f"dx={corr.get('rel_dx',0):.4f}"
                print(f"({M},{N},{K}){'':<8} {'FAIL':>8}   {err[:30]}")
            continue

        bench = benchmark_config(M, K, N, warmup=warmup, iters=iters)
        results.append({"config": (M, N, K), "correct": True, **bench})

        if verbose:
            marker = " *" if bench["speedup"] > 1.0 else ""
            print(
                f"({M},{N},{K}){'':<8} {'PASS':>8} "
                f"{bench['torch_ms']:>8.3f}ms {bench['triton_ms']:>8.3f}ms "
                f"{bench['speedup']:>8.2f}x{marker}"
            )

    speedups = [r["speedup"] for r in results]
    geomean = math.prod(speedups) ** (1 / len(speedups)) if speedups else 0.0

    if verbose:
        print("-" * 62)
        status = "ALL PASS" if not failures else f"FAILED ({len(failures)}/{len(configs)})"
        print(f"{'Status:':<22} {status}")
        if speedups:
            print(f"{'Speedup (geomean):':<22} {geomean:.2f}x")

    return {
        "correct": len(failures) == 0,
        "num_correct": len(results),
        "num_failed": len(failures),
        "failures": failures,
        "results": results,
        "speedup_geomean": geomean,
    }


def run_profile(configs=None, warmup=3, iters=1, verbose=True):
    configs = configs or PROFILE_CONFIGS
    if verbose:
        print(f"Profile: {len(configs)} config(s)")

    for M, N, K in configs:
        x, w_up, w_down, dy = get_inputs(M, K, N)
        y, h0, h1, a, g = ff_fused_gated_forward(x, w_up, w_down, ACTIVATION)

        for _ in range(warmup):
            ff_fused_gated_backward_triton(dy, x, w_up, w_down, h0, h1, a, g, ACTIVATION)
        torch.cuda.synchronize()

        for _ in range(iters):
            ff_fused_gated_backward_triton(dy, x, w_up, w_down, h0, h1, a, g, ACTIVATION)
        torch.cuda.synchronize()

        if verbose:
            print(f"  ({M},{N},{K}) done")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FF Backward Kernel (Pure Triton)")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    print("=" * 62)
    print("Fused Gated MLP Backward — Pure Triton")
    print("=" * 62)

    if args.profile:
        print("\n[Profile Mode]")
        run_profile()
    else:
        print("\n[Evaluation]")
        evaluate()

    print("=" * 62)
