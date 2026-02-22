import torch
import triton
import triton.language as tl


@triton.jit
def _correct_attn_cp_out_kernel(
    outputs_ptr,
    new_output_ptr,
    lses_ptr,
    vlse_ptr,
    outputs_stride_B,
    outputs_stride_H,
    outputs_stride_D,
    lses_stride_N,
    lses_stride_B,
    lses_stride_H,
    lse_idx,
    HEAD_DIM: tl.constexpr,
    N_ROUNDED: tl.constexpr,
    IS_BASE_E: tl.constexpr,
):
    """
    Apply the all-gathered lses to correct each local rank's attention
    output. Computes a logsumexp-based correction factor and rescales
    the attention output.

    Args:
        outputs_ptr: Pointer to input tensor of shape [B, H, D]
        new_output_ptr: Pointer to output tensor of shape [B, H, D]
        lses_ptr: Pointer to input tensor of shape [N, B, H]
        vlse_ptr: Pointer to output tensor of shape [B, H]
    """
    batch_idx = tl.program_id(axis=0).to(tl.int64)
    head_idx = tl.program_id(axis=1).to(tl.int64)
    d_offsets = tl.arange(0, HEAD_DIM)
    num_n_offsets = tl.arange(0, N_ROUNDED)

    # shape = [N]
    lse_offsets = (
        num_n_offsets * lses_stride_N
        + batch_idx * lses_stride_B
        + head_idx * lses_stride_H
    )

    # calc final lse
    lse = tl.load(lses_ptr + lse_offsets)
    lse = tl.where((lse != lse) | (lse == float("inf")), -float("inf"), lse)
    lse_max = tl.max(lse, axis=0)
    lse_max = tl.where(lse_max == -float("inf"), 0, lse_max)
    lse -= lse_max
    if IS_BASE_E:
        lse_exp = tl.exp(lse)
        lse_acc = tl.sum(lse_exp, axis=0)
        lse = tl.log(lse_acc)
    else:
        lse_exp = tl.exp2(lse)
        lse_acc = tl.sum(lse_exp, axis=0)
        lse = tl.log2(lse_acc)
    lse += lse_max

    lse_offsets = batch_idx * lses_stride_B + head_idx * lses_stride_H
    tl.store(vlse_ptr + lse_offsets, lse)

    # shape = [D]
    output_offsets = (
        batch_idx * outputs_stride_B
        + head_idx * outputs_stride_H
        + d_offsets * outputs_stride_D
    )

    # correct output
    lse_offset = (
        lse_idx * lses_stride_N + batch_idx * lses_stride_B + head_idx * lses_stride_H
    )
    lse_tmp = tl.load(lses_ptr + lse_offset)
    lse_finally = lse_tmp - lse
    lse_finally = tl.where(
        (lse_finally != lse_finally) | (lse_finally == float("inf")),
        -float("inf"),
        lse_finally,
    )
    factor = tl.exp(lse_finally) if IS_BASE_E else tl.exp2(lse_finally)
    output = tl.load(outputs_ptr + output_offsets)
    output = output * factor

    tl.store(new_output_ptr + output_offsets, output)


def correct_attn_cp_out(
    out: torch.Tensor,
    lses: torch.Tensor,
    lse_idx: int,
    is_base_e: bool = True,
) -> tuple:
    """Correct the attention output using logsumexp from context parallelism.

    This is the standalone version that takes tensors directly (no distributed ops).

    Args:
        out: Tensor of shape [B, H, D] - attention output to correct
        lses: Tensor of shape [N, B, H] - log-sum-exp values from N ranks
        lse_idx: Index of the current rank in the lses tensor
        is_base_e: If True, use natural log/exp; if False, use log2/exp2

    Returns:
        Tuple of (corrected_out [B, H, D], final_lse [B, H])
    """
    # Normalize to 3D views
    if out.ndim == 4 and out.shape[1] == 1:
        out = out.squeeze(1)
    assert out.ndim == 3, f"expected out [B,H,D] or [B,1,H,D], got {tuple(out.shape)}"

    if lses.ndim == 4 and lses.shape[-1] == 1:
        lses = lses.squeeze(-1)
    if lses.ndim == 4 and lses.shape[1] == 1:
        lses = lses.squeeze(1)
    assert lses.ndim == 3, (
        f"expected lses [N,B,H] (optionally with a 1-sized extra dim), "
        f"got {tuple(lses.shape)}"
    )

    B, H, D = out.shape
    N = lses.shape[0]

    o_sB, o_sH, o_sD = out.stride()
    l_sN, l_sB, l_sH = lses.stride()

    # Allocate output lse with same B/H strides as lses
    lse = torch.empty_strided(
        (B, H), (l_sB, l_sH), device=lses.device, dtype=lses.dtype
    )

    new_out = torch.empty_like(out)

    grid = (B, H, 1)

    _correct_attn_cp_out_kernel[grid](
        out,
        new_out,
        lses,
        lse,
        o_sB,
        o_sH,
        o_sD,
        l_sN,
        l_sB,
        l_sH,
        lse_idx,
        HEAD_DIM=D,
        N_ROUNDED=N,
        IS_BASE_E=is_base_e,
    )
    return new_out, lse
