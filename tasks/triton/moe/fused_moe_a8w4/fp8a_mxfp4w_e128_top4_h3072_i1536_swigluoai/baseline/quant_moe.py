# Vendored from ROCm/aiter (commit 652f2d4b768239aefd515c929e84817681ff598d, MIT, see LICENSE):
#   aiter/ops/triton/_triton_kernels/moe/quant_moe.py  (_compute_static_fp8_quant, _downcast_to_static_fp8)
# The upstream file carries no copyright header, so the repository LICENSE applies.
# Changes: none to the kernels; the MXFP quantization kernels of the file are not used by this task.

import triton
import triton.language as tl


@triton.jit
def _compute_static_fp8_quant(tensor, scale):
    tensor = tensor.to(tl.float32)
    tensor = tensor / scale
    tensor = tensor.to(tl.float8e4nv)
    return tensor


@triton.jit
def _downcast_to_static_fp8(
    x_ptr,
    stride_x_m,
    stride_x_n,
    y_ptr,
    stride_y_m,
    stride_y_n,
    scale_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):

    x_dtype: tl.constexpr = x_ptr.dtype.element_ty
    tl.static_assert(
        (x_dtype == tl.bfloat16) or (x_dtype == tl.float16) or (x_dtype == tl.float32),
        f"{x_dtype=} must be bfloat16 or float16 or float32",
    )

    pid_m = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)

    start_m = pid_m * BLOCK_M
    start_n = pid_n * BLOCK_N

    x_ptr += start_m * stride_x_m + start_n * stride_x_n
    y_ptr += start_m * stride_y_m + start_n * stride_y_n

    offs_m = tl.arange(0, BLOCK_M)[None, :].to(tl.int64)
    offs_n = tl.arange(0, BLOCK_N)[:, None].to(tl.int64)

    mask_m = start_m + offs_m < M
    mask_n = start_n + offs_n < N
    mask_xy = mask_m & mask_n

    offs_x = offs_m * stride_x_m + offs_n * stride_x_n
    offs_y = offs_m * stride_y_m + offs_n * stride_y_n

    x = tl.load(x_ptr + offs_x, mask=mask_xy)

    y = _compute_static_fp8_quant(x, tl.load(scale_ptr))

    tl.store(y_ptr + offs_y, y, mask=mask_xy)
