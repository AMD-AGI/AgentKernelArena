"""Triton kernel: kda_gate_fwd_kernel — KDA gate with softplus activation."""
import torch
import triton
import triton.language as tl
from math import ceil


@triton.autotune(
    configs=[
        triton.Config({"BT": bt}, num_warps=nw, num_stages=ns)
        for bt in [32, 64, 128]
        for nw in [4, 8, 16]
        for ns in [2, 3]
    ],
    key=["H", "D"],
)
@triton.jit
def kda_gate_fwd_kernel(
    g, A, y, g_bias,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    T, H,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    i_t, i_h = tl.program_id(0), tl.program_id(1)
    n_t = i_t * BT

    b_a = tl.load(A + i_h).to(tl.float32)
    b_a = -tl.exp(b_a)

    stride_row = H * D
    stride_col = 1

    g_ptr = tl.make_block_ptr(
        base=g + i_h * D,
        shape=(T, D),
        strides=(stride_row, stride_col),
        offsets=(n_t, 0),
        block_shape=(BT, BD),
        order=(1, 0),
    )
    y_ptr = tl.make_block_ptr(
        base=y + i_h * D,
        shape=(T, D),
        strides=(stride_row, stride_col),
        offsets=(n_t, 0),
        block_shape=(BT, BD),
        order=(1, 0),
    )

    b_g = tl.load(g_ptr, boundary_check=(0, 1)).to(tl.float32)

    if HAS_BIAS:
        n_d = tl.arange(0, BD)
        bias_mask = n_d < D
        b_bias = tl.load(g_bias + i_h * D + n_d, mask=bias_mask, other=0.0).to(tl.float32)
        b_g = b_g + b_bias[None, :]

    g_scaled = b_g * beta
    use_linear = g_scaled > threshold
    sp = tl.where(use_linear, b_g, (1.0 / beta) * tl.log(1.0 + tl.exp(g_scaled)))
    b_y = b_a * sp

    tl.store(y_ptr, b_y.to(y.dtype.element_ty), boundary_check=(0, 1))


def fused_kda_gate(
    g: torch.Tensor,
    A: torch.Tensor,
    head_k_dim: int,
    g_bias: torch.Tensor = None,
    beta_val: float = 1.0,
    threshold: float = 20.0,
) -> torch.Tensor:
    """Forward pass for KDA gate with softplus.

    Args:
        g: [..., H*D] — gate input
        A: [H] — per-head scaling
        head_k_dim: int — dimension per head
        g_bias: [H*D] or None
        beta_val: softplus beta
        threshold: softplus threshold
    Returns:
        y: [..., H, D]
    """
    orig_shape = g.shape[:-1]
    g = g.view(-1, g.shape[-1])
    T = g.shape[0]
    HD = g.shape[1]
    H = A.numel()
    assert H * head_k_dim == HD

    y = torch.empty_like(g, dtype=torch.float32)

    def _next_power_of_2(n):
        n -= 1
        n |= n >> 1
        n |= n >> 2
        n |= n >> 4
        n |= n >> 8
        n |= n >> 16
        return n + 1

    def grid(meta):
        return (ceil(T / meta["BT"]), H)

    kda_gate_fwd_kernel[grid](
        g, A, y, g_bias, beta_val, threshold,
        T, H, head_k_dim,
        BD=_next_power_of_2(head_k_dim),
        HAS_BIAS=g_bias is not None,
    )

    y = y.view(*orig_shape, H, head_k_dim)
    return y
