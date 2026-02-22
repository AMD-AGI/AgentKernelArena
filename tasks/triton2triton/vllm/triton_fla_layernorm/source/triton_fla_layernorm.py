"""Triton kernel: layer_norm_fwd_kernel — LayerNorm with optional SiLU gating (z branch)."""
import torch
import triton
import triton.language as tl
from math import ceil


@triton.heuristics(
    {
        "HAS_BIAS": lambda args: args["B"] is not None,
        "HAS_Z": lambda args: args["Z"] is not None,
    }
)
@triton.jit
def layer_norm_fwd_kernel(
    X, Y, W, B, Z, Mean, Rstd,
    stride_x_row, stride_y_row, stride_z_row,
    M,
    N: tl.constexpr,
    eps,
    BLOCK_N: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
):
    row_start = tl.program_id(0) * ROWS_PER_BLOCK
    group = tl.program_id(1)

    rows = row_start + tl.arange(0, ROWS_PER_BLOCK)
    cols = tl.arange(0, BLOCK_N)

    row_offsets = rows[:, None] * stride_x_row
    col_offsets = cols[None, :] + group * N

    X_base = X + row_offsets + col_offsets
    Y_base = Y + rows[:, None] * stride_y_row + col_offsets

    row_mask = rows[:, None] < M
    col_mask = cols[None, :] < N
    mask = row_mask & col_mask

    x = tl.load(X_base, mask=mask, other=0.0).to(tl.float32)

    if HAS_Z and not NORM_BEFORE_GATE:
        Z_base = Z + rows[:, None] * stride_z_row + col_offsets
        z = tl.load(Z_base, mask=mask, other=0.0).to(tl.float32)
        x *= z * tl.sigmoid(z)

    if not IS_RMS_NORM:
        mean = tl.sum(x, axis=1) / N
        mean_offsets = group * M + rows
        mean_mask = rows < M
        tl.store(Mean + mean_offsets, mean, mask=mean_mask)
        xbar = tl.where(mask, x - mean[:, None], 0.0)
        var = tl.sum(xbar * xbar, axis=1) / N
    else:
        xbar = tl.where(mask, x, 0.0)
        var = tl.sum(xbar * xbar, axis=1) / N
        mean = 0.0

    rstd = tl.rsqrt(var + eps)

    rstd_offsets = group * M + rows
    rstd_mask = rows < M
    tl.store(Rstd + rstd_offsets, rstd, mask=rstd_mask)

    w_offsets = cols + group * N
    w_mask = cols < N
    w = tl.load(W + w_offsets, mask=w_mask, other=0.0).to(tl.float32)

    if HAS_BIAS:
        b = tl.load(B + w_offsets, mask=w_mask, other=0.0).to(tl.float32)

    if not IS_RMS_NORM:
        x_hat = (x - mean[:, None]) * rstd[:, None]
    else:
        x_hat = x * rstd[:, None]

    y = x_hat * w[None, :] + b[None, :] if HAS_BIAS else x_hat * w[None, :]

    if HAS_Z and NORM_BEFORE_GATE:
        Z_base = Z + rows[:, None] * stride_z_row + col_offsets
        z = tl.load(Z_base, mask=mask, other=0.0).to(tl.float32)
        y *= z * tl.sigmoid(z)

    tl.store(Y_base, y, mask=mask)


def layer_norm_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    eps: float = 1e-5,
    z: torch.Tensor = None,
    norm_before_gate: bool = True,
    is_rms_norm: bool = False,
) -> tuple:
    """Layer norm / RMS norm with optional SiLU gating.

    Args:
        x: [M, N]
        weight: [N]
        bias: [N] or None
        eps: float
        z: [M, N] or None — gate branch
        norm_before_gate: bool
        is_rms_norm: bool
    Returns:
        out, mean, rstd
    """
    M, N = x.shape
    out = torch.empty_like(x)
    mean = torch.empty((M,), dtype=torch.float32, device=x.device) if not is_rms_norm else None
    rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
    BLOCK_N = min(65536 // x.element_size(), triton.next_power_of_2(N))
    num_warps = min(max(BLOCK_N // 256, 1), 8)

    def _next_power_of_2(n):
        n -= 1
        n |= n >> 1
        n |= n >> 2
        n |= n >> 4
        n |= n >> 8
        n |= n >> 16
        return n + 1

    # Simple rows_per_block heuristic
    rows_per_block = min(_next_power_of_2(ceil(M / 128)), 4)
    rows_per_block = max(rows_per_block, 1)

    grid = (ceil(M / rows_per_block), 1)
    layer_norm_fwd_kernel[grid](
        x, out, weight, bias, z,
        mean, rstd,
        x.stride(0), out.stride(0),
        z.stride(0) if z is not None else 0,
        M, N, eps,
        BLOCK_N=BLOCK_N,
        ROWS_PER_BLOCK=rows_per_block,
        NORM_BEFORE_GATE=norm_before_gate,
        IS_RMS_NORM=is_rms_norm,
        num_warps=num_warps,
    )
    return out, mean, rstd
