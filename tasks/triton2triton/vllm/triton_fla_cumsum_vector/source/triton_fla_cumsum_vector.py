"""Triton kernel: chunk_local_cumsum_vector_kernel — chunk-local cumulative sum for vector gates."""
import torch
import triton
import triton.language as tl
from math import ceil


@triton.autotune(
    configs=[
        triton.Config({"BS": BS}, num_warps=num_warps)
        for BS in [32, 64]
        for num_warps in [2, 4, 8]
    ],
    key=["B", "H", "S", "BT"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_local_cumsum_vector_kernel(
    s, o,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    REVERSE: tl.constexpr,
):
    i_s, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b = i_bh // H
    i_h = i_bh % H
    bos = i_b * T

    o_i = tl.arange(0, BT)
    if REVERSE:
        m_s = tl.where(o_i[:, None] <= o_i[None, :], 1.0, 0.0)
    else:
        m_s = tl.where(o_i[:, None] >= o_i[None, :], 1.0, 0.0)

    p_s = tl.make_block_ptr(
        s + (bos * H + i_h) * S, (T, S), (H * S, 1),
        (i_t * BT, i_s * BS), (BT, BS), (1, 0),
    )
    p_o = tl.make_block_ptr(
        o + (bos * H + i_h) * S, (T, S), (H * S, 1),
        (i_t * BT, i_s * BS), (BT, BS), (1, 0),
    )
    b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
    b_o = tl.dot(m_s, b_s, allow_tf32=False)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def chunk_local_cumsum_vector(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
) -> torch.Tensor:
    """Chunk-local cumulative sum for vector gates.

    Args:
        g: [B, T, H, S]
        chunk_size: int (power of 2)
        reverse: bool
    Returns:
        output: [B, T, H, S] with cumsum within each chunk
    """
    B, T, H, S = g.shape
    BT = chunk_size
    NT = ceil(T / BT)
    out = torch.empty_like(g, dtype=torch.float32)

    def grid(meta):
        return (ceil(S / meta["BS"]), NT, B * H)

    chunk_local_cumsum_vector_kernel[grid](
        g, out, T=T, B=B, H=H, S=S, BT=BT, REVERSE=reverse,
    )
    return out
