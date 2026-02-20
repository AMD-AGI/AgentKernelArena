"""Triton kernel: chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter — inter-sub-block KDA dot product."""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BK": BK}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64]
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["BC"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter(
    q, k, g, beta, A, Aqk, scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
):
    i_t, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b = i_bh // H
    i_h = i_bh % H
    i_i = i_c // NC
    i_j = i_c % NC

    bos = i_b * T

    if i_t * BT + i_i * BC >= T:
        return
    if i_i <= i_j:
        return

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += (bos * H + i_h) * K
    A += (bos * H + i_h) * BT
    Aqk += (bos * H + i_h) * BT

    p_b = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_t * BT + i_i * BC,), (BC,), (0,))
    b_b = tl.load(p_b, boundary_check=(0,))

    b_A = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk = tl.zeros([BC, BC], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_g = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        b_kt = tl.make_block_ptr(k, (K, T), (1, H * K), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
        p_gk = tl.make_block_ptr(g, (K, T), (1, H * K), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))

        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        b_gn = tl.load(g + (i_t * BT + i_i * BC) * H * K + o_k, mask=m_k, other=0)
        b_g_val = tl.load(p_g, boundary_check=(0, 1))
        b_k_val = tl.load(p_k, boundary_check=(0, 1)) * tl.exp(b_g_val - b_gn[None, :])
        b_gk_val = tl.load(p_gk, boundary_check=(0, 1))
        b_kt_val = tl.load(b_kt, boundary_check=(0, 1))
        b_ktg = b_kt_val * tl.exp(b_gn[:, None] - b_gk_val)
        b_A += tl.dot(b_k_val, b_ktg)

        b_q_val = tl.load(p_q, boundary_check=(0, 1))
        b_qg = b_q_val * tl.exp(b_g_val - b_gn[None, :]) * scale
        b_Aqk += tl.dot(b_qg, b_ktg)

    b_A *= b_b[:, None]

    p_A = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
    tl.store(p_A, b_A.to(A.dtype.element_ty), boundary_check=(0, 1))
    p_Aqk = tl.make_block_ptr(Aqk, (T, BT), (H * BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
    tl.store(p_Aqk, b_Aqk.to(Aqk.dtype.element_ty), boundary_check=(0, 1))


def kda_dot_kkt_inter(
    q: torch.Tensor,
    k: torch.Tensor,
    gk: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    chunk_size: int = 64,
) -> tuple:
    """Compute inter-sub-block part of beta * K @ K^T with gating for KDA.

    Args:
        q: [B, T, H, K]
        k: [B, T, H, K]
        gk: [B, T, H, K] — cumulative gate
        beta: [B, T, H]
        scale: float
        chunk_size: int
    Returns:
        A: [B, T, H, BT], Aqk: [B, T, H, BT]
    """
    from math import ceil
    B, T, H, K = k.shape
    BT = chunk_size
    NT = ceil(T / BT)
    BC = min(16, BT)
    NC = ceil(BT / BC)
    A = torch.zeros(B, T, H, BT, device=k.device, dtype=torch.float32)
    Aqk = torch.zeros(B, T, H, BT, device=k.device, dtype=torch.float32)
    grid = (NT, NC * NC, B * H)
    chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter[grid](
        q=q, k=k, g=gk, beta=beta, A=A, Aqk=Aqk, scale=scale,
        T=T, H=H, K=K, BT=BT, BC=BC, NC=NC,
    )
    return A, Aqk
