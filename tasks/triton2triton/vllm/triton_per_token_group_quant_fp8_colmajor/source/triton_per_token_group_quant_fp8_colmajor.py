import torch
import triton
import triton.language as tl


def _get_fp8_dtype():
    """Get the appropriate FP8 dtype for the current platform."""
    if hasattr(torch, 'float8_e4m3fnuz'):
        try:
            t = torch.zeros(1, device='cuda', dtype=torch.float8_e4m3fnuz)
            return torch.float8_e4m3fnuz
        except Exception:
            pass
    return torch.float8_e4m3fn


def _get_fp8_min_max():
    """Get FP8 min/max values for the current platform."""
    fp8_dtype = _get_fp8_dtype()
    if fp8_dtype == torch.float8_e4m3fnuz:
        return -240.0, 240.0
    else:
        finfo = torch.finfo(fp8_dtype)
        return finfo.min, finfo.max


@triton.jit
def _per_token_group_quant_fp8_colmajor(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    y_num_columns,
    y_row_stride,
    y_s_col_stride,
    eps,
    fp8_min,
    fp8_max,
    use_ue8m0: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """A Triton-accelerated function to perform per-token-group
    quantization on a tensor with column-major scale output.
    This function converts the tensor values into float8 values.
    """
    groups_per_row = y_num_columns // group_size

    g_id = tl.program_id(0)
    row = g_id // groups_per_row
    row_g_id = g_id % groups_per_row

    y_ptr_offset = (row.to(tl.int64) * y_row_stride) + (
        row_g_id.to(tl.int64) * group_size
    )
    y_ptr += y_ptr_offset

    y_q_ptr_offset = g_id.to(tl.int64) * group_size
    y_q_ptr += y_q_ptr_offset

    blocks_per_row = y_num_columns // group_size
    scale_col = g_id % blocks_per_row
    scale_row = g_id // blocks_per_row
    y_s_ptr_offset = (scale_col.to(tl.int64) * y_s_col_stride) + scale_row.to(tl.int64)
    y_s_ptr += y_s_ptr_offset

    cols = tl.arange(0, BLOCK)
    mask = cols < group_size

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    scale_raw = _absmax / fp8_max
    y_s = tl.math.exp2(tl.ceil(tl.log2(scale_raw))) if use_ue8m0 else scale_raw
    y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


def per_token_group_quant_fp8_colmajor(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: torch.dtype | None = None,
    use_ue8m0: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Function to perform per-token-group quantization with column-major scales."""
    if dtype is None:
        dtype = _get_fp8_dtype()

    assert x.shape[-1] % group_size == 0
    assert x.stride(-1) == 1

    fp8_min, fp8_max = _get_fp8_min_max()

    x_q = torch.empty(x.shape, device=x.device, dtype=dtype)

    # Column-major scales
    shape = (x.shape[-1] // group_size, x.shape[-2])
    x_s = torch.empty(shape, device=x.device, dtype=torch.float32).permute(-1, -2)

    M = x.numel() // group_size
    N = group_size
    BLOCK = triton.next_power_of_2(N)
    num_warps = min(max(BLOCK // 256, 1), 8)
    num_stages = 1

    _per_token_group_quant_fp8_colmajor[(M,)](
        x,
        x_q,
        x_s,
        group_size,
        x.shape[1],
        x.stride(0),
        x_s.stride(1),
        eps,
        fp8_min=fp8_min,
        fp8_max=fp8_max,
        use_ue8m0=use_ue8m0,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return x_q, x_s
