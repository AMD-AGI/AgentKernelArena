"""Triton kernel: merge_16x16_to_32x32_inverse_kernel — merge 16x16 inverses into 32x32."""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4, 5]
    ],
    key=["H", "BT"],
)
@triton.jit(do_not_specialize=["T"])
def merge_16x16_to_32x32_inverse_kernel(
    A,
    Ai,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H
    bos = i_b * T

    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    A += (bos * H + i_h) * BT
    Ai += (bos * H + i_h) * BT

    p_A_11 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT, 0), (16, 16), (1, 0))
    p_A_22 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0))
    b_Ai_11 = tl.load(p_A_11, boundary_check=(0, 1)).to(tl.float32)
    b_Ai_22 = tl.load(p_A_22, boundary_check=(0, 1)).to(tl.float32)

    b_Ai_11 = -tl.where(m_A, b_Ai_11, 0)
    b_Ai_22 = -tl.where(m_A, b_Ai_22, 0)

    for i in range(2, min(16, T - i_t * BT)):
        b_a_11 = -tl.load(A + (i_t * BT + i) * H * BT + o_i)
        b_a_11 += tl.sum(b_a_11[:, None] * b_Ai_11, 0)
        b_Ai_11 = tl.where((o_i == i)[:, None], b_a_11, b_Ai_11)
    for i in range(16 + 2, min(32, T - i_t * BT)):
        b_a_22 = -tl.load(A + (i_t * BT + i) * H * BT + o_i + 16)
        b_a_22 += tl.sum(b_a_22[:, None] * b_Ai_22, 0)
        b_Ai_22 = tl.where((o_i == i - 16)[:, None], b_a_22, b_Ai_22)

    b_Ai_11 += m_I
    b_Ai_22 += m_I

    p_A_21 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0))
    b_A_21 = tl.load(p_A_21, boundary_check=(0, 1)).to(tl.float32)

    b_Ai_21 = -tl.dot(
        tl.dot(b_Ai_22, b_A_21, input_precision=DOT_PRECISION),
        b_Ai_11,
        input_precision=DOT_PRECISION,
    )

    p_Ai_11 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT, 0), (16, 16), (1, 0))
    p_Ai_21 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0))
    p_Ai_22 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0))
    tl.store(p_Ai_11, b_Ai_11.to(p_Ai_11.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai_22, b_Ai_22.to(p_Ai_22.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai_21, b_Ai_21.to(p_Ai_21.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))


def merge_16x16_to_32x32(A: torch.Tensor) -> torch.Tensor:
    """Compute (I + A)^{-1} for BT=32 blocks.

    Args:
        A: [B, T, H, 32] — strictly lower triangular blocks
    Returns:
        Ai: [B, T, H, 32] — inverse blocks
    """
    B, T, H, BT = A.shape
    assert BT == 32
    from math import ceil
    NT = ceil(T / BT)
    Ai = torch.zeros_like(A, dtype=torch.float32)
    merge_16x16_to_32x32_inverse_kernel[(NT, B * H)](
        A=A,
        Ai=Ai,
        T=T,
        H=H,
        BT=BT,
        DOT_PRECISION="ieee",
    )
    return Ai
