"""Triton kernel: chunk_local_cumsum_scalar_kernel — chunk-local cumulative sum for scalar gates."""
import torch
import triton
import triton.language as tl
from math import ceil


@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4, 8]],
    key=["B", "H", "BT"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_local_cumsum_scalar_kernel(
    s, o,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    REVERSE: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H
    bos = i_b * T

    p_s = tl.make_block_ptr(s + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_o = tl.make_block_ptr(o + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_s = tl.load(p_s, boundary_check=(0,)).to(tl.float32)
    b_o = tl.cumsum(b_s, axis=0)
    if REVERSE:
        b_z = tl.sum(b_s, axis=0)
        b_o = -b_o + b_z[None] + b_s
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))


def chunk_local_cumsum_scalar(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
) -> torch.Tensor:
    """Chunk-local cumulative sum for scalar gates.

    Args:
        g: [B, T, H]
        chunk_size: int (power of 2)
        reverse: bool
    Returns:
        output: [B, T, H] with cumsum within each chunk
    """
    B, T, H = g.shape
    BT = chunk_size
    NT = ceil(T / BT)
    out = torch.empty_like(g, dtype=torch.float32)
    grid = (NT, B * H)
    chunk_local_cumsum_scalar_kernel[grid](
        g, out, T=T, B=B, H=H, BT=BT, REVERSE=reverse,
    )
    return out
