"""Triton kernel: recompute_w_u_fwd_kernel (KDA variant) — recompute w and u from A, k, v, beta, gk."""
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
    k, v, beta, w, u, A, gk,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H
    bos = i_b * T

    p_b = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_b = tl.load(p_b, boundary_check=(0,))

    p_A = tl.make_block_ptr(A + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_A = tl.load(p_A, boundary_check=(0, 1))

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_u = tl.make_block_ptr(u + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, input_precision=DOT_PRECISION)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))

    for i_k in range(tl.cdiv(K, BK)):
        p_w = tl.make_block_ptr(w + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = b_k * b_b[:, None]

        p_gk = tl.make_block_ptr(gk + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_gk = tl.load(p_gk, boundary_check=(0, 1))
        b_kb *= tl.exp(b_gk)

        b_w = tl.dot(b_A, b_kb.to(b_k.dtype))
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))


def kda_recompute_wu(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    gk: torch.Tensor,
) -> tuple:
    """Recompute w and u for KDA forward pass.

    Args:
        k: [B, T, H, K]
        v: [B, T, H, V]
        beta: [B, T, H]
        A: [B, T, H, BT] — solve_tril result
        gk: [B, T, H, K] — cumulative gate
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
    w = torch.empty_like(k)
    u = torch.empty_like(v)
    recompute_w_u_fwd_kernel[(NT, B * H)](
        k=k, v=v, beta=beta, w=w, u=u, A=A, gk=gk,
        T=T, H=H, K=K, V=V, BT=BT, BK=BK, BV=BV,
        DOT_PRECISION="ieee",
    )
    return w, u
