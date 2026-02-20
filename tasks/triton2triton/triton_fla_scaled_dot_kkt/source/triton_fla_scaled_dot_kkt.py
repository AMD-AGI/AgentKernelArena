"""Triton kernel: chunk_scaled_dot_kkt_fwd_kernel — Compute beta * K @ K^T with optional gating."""
import torch
import triton
import triton.language as tl


@triton.heuristics({"USE_G": lambda args: args["g"] is not None})
@triton.autotune(
    configs=[
        triton.Config({"BK": BK}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64, 128]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["H", "K", "BT"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_scaled_dot_kkt_fwd_kernel(
    k, beta, g, A,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    USE_G: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H
    bos = i_b * T

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T

    p_beta = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_beta = tl.load(p_beta, boundary_check=(0,))

    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(
            k + (bos * H + i_h) * K, (T, K), (H * K, 1),
            (i_t * BT, i_k * BK), (BT, BK), (1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = b_k * b_beta[:, None]
        b_A += tl.dot(b_kb.to(b_k.dtype), tl.trans(b_k))

    if USE_G:
        p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_g_diff = b_g[:, None] - b_g[None, :]
        b_A = b_A * tl.exp(b_g_diff)

    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)
    p_A = tl.make_block_ptr(
        A + (bos * H + i_h) * BT, (T, BT), (BT * H, 1),
        (i_t * BT, 0), (BT, BT), (1, 0),
    )
    tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))


def chunk_scaled_dot_kkt_fwd(
    k: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Compute beta * K @ K^T within chunks.

    Args:
        k: [B, T, H, K]
        beta: [B, T, H]
        g: [B, T, H] or None — cumulative gate
        chunk_size: int
    Returns:
        A: [B, T, H, BT]
    """
    from math import ceil
    B, T, H, K = k.shape
    BT = chunk_size
    NT = ceil(T / BT)
    A = torch.empty(B, T, H, BT, device=k.device, dtype=torch.float32)
    chunk_scaled_dot_kkt_fwd_kernel[(NT, B * H)](
        k=k, g=g, beta=beta, A=A,
        T=T, H=H, K=K, BT=BT,
    )
    return A
