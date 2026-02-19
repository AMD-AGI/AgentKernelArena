import torch
import triton
import triton.language as tl


@triton.jit
def triton_scale_swizzle(
    scale_ptr: torch.Tensor,
    scale_rows: int,
    scale_cols: int,
    output_ptr: torch.Tensor,
    input_row_stride: int,
    output_block_stride: int,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    """
    Rearranges tensor data from row-major to block-scaled swizzle format.
    """
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    rows = tl.arange(0, BLOCK_ROWS)[:, None]
    cols = tl.arange(0, BLOCK_COLS)[None, :]

    start_row = pid_row * BLOCK_ROWS
    start_col = pid_col * BLOCK_COLS
    global_rows = start_row + rows
    global_cols = start_col + cols

    mask = (global_rows < scale_rows) & (global_cols < scale_cols)

    input_scales = tl.load(
        scale_ptr + global_rows * input_row_stride + global_cols,
        mask=mask,
        other=0.0,
    )

    r_div_32 = rows // 32
    r_mod_32 = rows % 32

    dest_indices = r_mod_32 * 16 + r_div_32 * 4 + cols

    dest_indices_flat = tl.reshape(dest_indices, (BLOCK_ROWS * BLOCK_COLS))
    scales_flat = tl.reshape(input_scales, (BLOCK_ROWS * BLOCK_COLS))

    LOCAL_NUMEL = BLOCK_ROWS * BLOCK_COLS
    block_offset = pid_col * LOCAL_NUMEL + (pid_row * output_block_stride)

    tl.store(
        output_ptr + block_offset + dest_indices_flat,
        scales_flat,
    )


def cdiv(a, b):
    return (a + b - 1) // b


def triton_mx_block_rearrange(scale_tensor: torch.Tensor) -> torch.Tensor:
    """
    Rearranges an E8M0 tensor scale from row-major format to
    block-scaled swizzle format.
    """
    assert scale_tensor.element_size() == 1, (
        "Expected element size to be 1 byte (8 bits)"
    )
    assert scale_tensor.is_contiguous(), "Input tensor must be contiguous"

    rows, cols = scale_tensor.shape

    n_row_blocks = triton.cdiv(rows, 128)
    n_col_blocks = triton.cdiv(cols, 4)
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    out = scale_tensor.new_empty((padded_rows, padded_cols))

    input_row_stride = cols

    BLOCK_ROWS, BLOCK_COLS = 128, 4

    output_block_stride = BLOCK_ROWS * BLOCK_COLS * (padded_cols // BLOCK_COLS)

    grid = lambda META: (
        triton.cdiv(padded_rows, BLOCK_ROWS),
        triton.cdiv(padded_cols, BLOCK_COLS),
    )

    triton_scale_swizzle[grid](
        scale_tensor.view(torch.uint8),
        rows,
        cols,
        out.view(torch.uint8),
        input_row_stride,
        output_block_stride,
        BLOCK_ROWS=BLOCK_ROWS,
        BLOCK_COLS=BLOCK_COLS,
    )

    return out
