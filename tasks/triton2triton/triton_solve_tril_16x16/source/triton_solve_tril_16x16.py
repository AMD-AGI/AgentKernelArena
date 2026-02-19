"""Triton kernel: solve_tril_16x16_kernel — Compute (I+A)^{-1} for 16x16 lower-triangular blocks."""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4, 5]
    ],
    key=["BT"],
)
@triton.jit(do_not_specialize=["T"])
def solve_tril_16x16_kernel(
    A,
    Ai,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
):
    """Invert each 16x16 block of (I + A) where A is strictly lower triangular.

    Grid: (NT, B*H)
    A:  [B, T, H, BT]  — input strictly lower triangular blocks
    Ai: [B, T, H, 16]  — output inverse blocks (16x16 stored in last dim=16)
    """
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H

    bos = i_b * T

    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    A_base = A + (bos * H + i_h) * BT
    Ai_base = Ai + (bos * H + i_h) * 16

    offset = (i_t * 16) % BT
    p_A = tl.make_block_ptr(
        A_base, (T, BT), (H * BT, 1), (i_t * 16, offset), (16, 16), (1, 0)
    )
    b_A = tl.load(p_A, boundary_check=(0, 1)).to(tl.float32)
    b_A = -tl.where(m_A, b_A, 0)

    for i in range(2, min(16, T - i_t * 16)):
        b_a = -tl.load(A_base + (i_t * 16 + i) * H * BT + o_i + offset)
        b_a = b_a + tl.sum(b_a[:, None] * b_A, 0)
        b_A = tl.where((o_i == i)[:, None], b_a, b_A)
    b_A += m_I

    p_Ai = tl.make_block_ptr(
        Ai_base, (T, 16), (H * 16, 1), (i_t * 16, 0), (16, 16), (1, 0)
    )
    tl.store(
        p_Ai,
        b_A.to(p_Ai.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )


def solve_tril_16x16(A: torch.Tensor) -> torch.Tensor:
    """Compute (I + A)^{-1} for BT=16 blocks.

    Args:
        A: [B, T, H, 16] — strictly lower triangular blocks
    Returns:
        Ai: [B, T, H, 16] — inverse blocks
    """
    B, T, H, BT = A.shape
    assert BT == 16
    from math import ceil
    NT = ceil(T / BT)
    Ai = torch.zeros(B, T, H, 16, dtype=torch.float32, device=A.device)
    solve_tril_16x16_kernel[(NT, B * H)](
        A=A,
        Ai=Ai,
        T=T,
        H=H,
        BT=BT,
    )
    return Ai
