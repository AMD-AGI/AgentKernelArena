"""Triton kernel: recompute_w_u_fwd_kernel (wy_fast variant) — recompute w and u with scalar gate."""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["H", "K", "V", "BT", "BK", "BV"],
)
@triton.jit(do_not_specialize=["T"])
def recompute_w_u_fwd_kernel(
    k, v, beta, w, u, A, g,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H
    bos = i_b * T

    p_beta = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_g = tl.make_block_ptr(g + (bos * H + i_h), (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_A = tl.make_block_ptr(A + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_beta = tl.load(p_beta, boundary_check=(0,))
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_g = tl.exp(tl.load(p_g, boundary_check=(0,)))

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_u = tl.make_block_ptr(u + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, allow_tf32=False)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))

    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_w = tl.make_block_ptr(w + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = (b_k * b_beta[:, None] * b_g[:, None]).to(b_k.dtype)
        b_w = tl.dot(b_A, b_kb)
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))


def wy_fast_recompute_wu(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    A: torch.Tensor,
) -> tuple:
    """Recompute w and u for WY-fast forward pass.

    Args:
        k: [B, T, H, K]
        v: [B, T, H, V]
        beta: [B, T, H]
        g_cumsum: [B, T, H] — cumulative scalar gate
        A: [B, T, H, BT]
    Returns:
        w: [B, T, H, K], u: [B, T, H, V]
    """
    from math import ceil
    B, T, H, K = k.shape
    V = v.shape[-1]
    BT = A.shape[-1]
    NT = ceil(T / BT)
    BK = 64
    BV = 64
    u = torch.empty_like(v)
    w = torch.empty(B, T, H, K, device=k.device, dtype=k.dtype)
    recompute_w_u_fwd_kernel[(NT, B * H)](
        k=k, v=v, beta=beta, w=w, u=u, A=A, g=g_cumsum,
        T=T, H=H, K=K, V=V, BT=BT, BK=BK, BV=BV,
    )
    return w, u
