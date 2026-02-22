"""Triton kernel: chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra — intra-sub-block KDA dot product."""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4, 8]],
    key=["BK", "BT"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra(
    q, k, g, beta, A, Aqk, scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
):
    i_t, i_i, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b = i_bh // H
    i_h = i_bh % H
    bos = i_b * T

    if i_t * BT + i_i * BC >= T:
        return

    o_i = tl.arange(0, BC)
    o_k = tl.arange(0, BK)
    m_k = o_k < K
    m_A = (i_t * BT + i_i * BC + o_i) < T
    o_A = (bos + i_t * BT + i_i * BC + o_i) * H * BT + i_h * BT + i_i * BC

    p_q = tl.make_block_ptr(q + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT + i_i * BC, 0), (BC, BK), (1, 0))
    p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT + i_i * BC, 0), (BC, BK), (1, 0))
    p_g = tl.make_block_ptr(g + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT + i_i * BC, 0), (BC, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_g = tl.load(p_g, boundary_check=(0, 1))

    p_b = beta + (bos + i_t * BT + i_i * BC + o_i) * H + i_h
    b_k = b_k * tl.load(p_b, mask=m_A, other=0)[:, None]

    p_kt = k + (bos + i_t * BT + i_i * BC) * H * K + i_h * K + o_k
    p_gk = g + (bos + i_t * BT + i_i * BC) * H * K + i_h * K + o_k

    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        b_kt = tl.load(p_kt, mask=m_k, other=0).to(tl.float32)
        b_gk = tl.load(p_gk, mask=m_k, other=0).to(tl.float32)
        b_ktg = b_kt[None, :] * tl.exp(b_g - b_gk[None, :])
        b_A_val = tl.sum(b_k * b_ktg, 1)
        b_A_val = tl.where(o_i > j, b_A_val, 0.0)
        b_Aqk_val = tl.sum(b_q * b_ktg, 1)
        b_Aqk_val = tl.where(o_i >= j, b_Aqk_val * scale, 0.0)
        tl.store(A + o_A + j, b_A_val, mask=m_A)
        tl.store(Aqk + o_A + j, b_Aqk_val, mask=m_A)
        p_kt += H * K
        p_gk += H * K


def kda_dot_kkt_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    gk: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    chunk_size: int = 64,
) -> tuple:
    """Compute intra-sub-block part of beta * K @ K^T with gating for KDA.

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
    BK = max(triton.next_power_of_2(K), 16)
    A = torch.zeros(B, T, H, BT, device=k.device, dtype=torch.float32)
    Aqk = torch.zeros(B, T, H, BT, device=k.device, dtype=torch.float32)
    grid = (NT, NC, B * H)
    chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra[grid](
        q=q, k=k, g=gk, beta=beta, A=A, Aqk=Aqk, scale=scale,
        T=T, H=H, K=K, BT=BT, BC=BC, BK=BK,
    )
    return A, Aqk
