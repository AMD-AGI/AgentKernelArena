"""Triton kernel: l2norm_fwd_kernel — L2 normalization over the last dimension."""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BT": BT}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8, 16]
        for BT in [8, 16, 32, 64, 128]
    ],
    key=["D"],
)
@triton.jit(do_not_specialize=["NB"])
def l2norm_fwd_kernel(
    x, y, eps, NB,
    T,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    i_t = tl.program_id(0)
    p_x = tl.make_block_ptr(x, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)
    b_var = tl.sum(b_x * b_x, axis=1)
    b_y = b_x / tl.sqrt(b_var + eps)[:, None]
    p_y = tl.make_block_ptr(y, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))


def l2norm_fwd(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """L2 normalize x along the last dimension.

    Args:
        x: [..., D]
        eps: float
    Returns:
        y: [..., D] with ||y||_2 = 1
    """
    x_shape_og = x.shape
    x = x.view(-1, x.shape[-1])
    y = torch.empty_like(x)
    T, D = x.shape

    BD = min(65536 // x.element_size(), triton.next_power_of_2(D))
    NB = triton.cdiv(T, 2048)

    def grid(meta):
        return (triton.cdiv(T, meta["BT"]),)

    l2norm_fwd_kernel[grid](x, y, eps, NB=NB, T=T, D=D, BD=BD)
    return y.view(x_shape_og)
