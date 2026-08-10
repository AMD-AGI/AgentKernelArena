import torch
import triton
import triton.language as tl

AWQ_TRITON_SUPPORTED_GROUP_SIZES = (32, 64, 128)
AWQ_TRITON_MAX_BLOCK_SIZE = 128


@triton.jit
def awq_dequantize_kernel(
    qweight_ptr,
    scales_ptr,
    zeros_ptr,
    group_size,
    result_ptr,
    num_cols,
    num_rows,
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
):
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)

    offsets_y = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    offsets_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    offsets = num_cols * offsets_y[:, None] + offsets_x[None, :]

    masks_y = offsets_y < num_rows
    masks_x = offsets_x < num_cols
    masks = masks_y[:, None] & masks_x[None, :]

    result_offsets_y = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    result_offsets_x = pid_x * BLOCK_SIZE_X * 8 + tl.arange(0, BLOCK_SIZE_X * 8)
    result_offsets = (
        8 * num_cols * result_offsets_y[:, None] + result_offsets_x[None, :]
    )

    result_masks_y = result_offsets_y < num_rows
    result_masks_x = result_offsets_x < num_cols * 8
    result_masks = result_masks_y[:, None] & result_masks_x[None, :]

    iweights = tl.load(qweight_ptr + offsets, masks, 0.0)
    iweights = tl.interleave(iweights, iweights)
    iweights = tl.interleave(iweights, iweights)
    iweights = tl.interleave(iweights, iweights)

    reverse_awq_order_tensor = (
        (tl.arange(0, 2) * 4)[None, :] + tl.arange(0, 4)[:, None]
    ).reshape(8)

    shifts = reverse_awq_order_tensor * 4
    shifts = tl.broadcast_to(shifts[None, :], (BLOCK_SIZE_Y * BLOCK_SIZE_X, 8))
    shifts = tl.reshape(shifts, (BLOCK_SIZE_Y, BLOCK_SIZE_X * 8))

    iweights = (iweights >> shifts) & 0xF

    zero_offsets_y = pid_y * BLOCK_SIZE_Y // group_size + tl.arange(0, 1)
    zero_offsets_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    zero_offsets = num_cols * zero_offsets_y[:, None] + zero_offsets_x[None, :]

    zero_masks_y = zero_offsets_y < num_rows // group_size
    zero_masks_x = zero_offsets_x < num_cols
    zero_masks = zero_masks_y[:, None] & zero_masks_x[None, :]

    zeros = tl.load(zeros_ptr + zero_offsets, zero_masks, 0.0)
    zeros = tl.interleave(zeros, zeros)
    zeros = tl.interleave(zeros, zeros)
    zeros = tl.interleave(zeros, zeros)
    zeros = tl.broadcast_to(zeros, (BLOCK_SIZE_Y, BLOCK_SIZE_X * 8))

    zeros = (zeros >> shifts) & 0xF

    scale_offsets_y = pid_y * BLOCK_SIZE_Y // group_size + tl.arange(0, 1)
    scale_offsets_x = pid_x * BLOCK_SIZE_X * 8 + tl.arange(0, BLOCK_SIZE_X * 8)
    scale_offsets = num_cols * 8 * scale_offsets_y[:, None] + scale_offsets_x[None, :]
    scale_masks_y = scale_offsets_y < num_rows // group_size
    scale_masks_x = scale_offsets_x < num_cols * 8
    scale_masks = scale_masks_y[:, None] & scale_masks_x[None, :]

    scales = tl.load(scales_ptr + scale_offsets, scale_masks, 0.0)
    scales = tl.broadcast_to(scales, (BLOCK_SIZE_Y, BLOCK_SIZE_X * 8))

    iweights = (iweights - zeros) * scales
    iweights = iweights.to(result_ptr.type.element_ty)

    tl.store(result_ptr + result_offsets, iweights, result_masks)


def _validate_awq_dequantize_contract(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    block_size_x: int,
    block_size_y: int,
) -> tuple[int, int, int]:
    tensors = (qweight, scales, zeros)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError("qweight, scales, and zeros must be torch.Tensor instances")
    if not all(tensor.ndim == 2 for tensor in tensors):
        raise ValueError("qweight, scales, and zeros must be two-dimensional")
    if qweight.dtype != torch.int32 or zeros.dtype != torch.int32:
        raise TypeError("qweight and zeros must have dtype torch.int32")
    if scales.dtype != torch.float16:
        raise TypeError("scales must have dtype torch.float16")
    if not (qweight.device == scales.device == zeros.device):
        raise ValueError("qweight, scales, and zeros must be on one device")
    if qweight.device.type != "cuda":
        raise ValueError("AWQ dequantization requires a CUDA/ROCm device")
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise ValueError("all AWQ inputs must be C-contiguous; implicit copies are forbidden")

    K, num_cols = qweight.shape
    if K <= 0 or num_cols <= 0:
        raise ValueError("K and N_packed must be positive")
    num_groups = scales.shape[0]
    if num_groups <= 0 or K % num_groups != 0:
        raise ValueError("the scales row count must divide K exactly")
    group_size = K // num_groups
    if (
        group_size not in AWQ_TRITON_SUPPORTED_GROUP_SIZES
        and group_size != K
    ):
        raise ValueError("group_size must be 32, 64, 128, or K")
    if scales.shape != (num_groups, num_cols * 8):
        raise ValueError("scales must have shape [K/group_size, N_packed*8]")
    if zeros.shape != (num_groups, num_cols):
        raise ValueError("zeros must have shape [K/group_size, N_packed]")

    for name, value in (
        ("block_size_x", block_size_x),
        ("block_size_y", block_size_y),
    ):
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
            or value > AWQ_TRITON_MAX_BLOCK_SIZE
            or value & (value - 1)
        ):
            raise ValueError(f"{name} must be a positive power of two no larger than 128")
    if group_size % block_size_y != 0:
        raise ValueError("block_size_y must divide group_size")
    return K, num_cols, group_size


def awq_dequantize_triton(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    block_size_x: int = 32,
    block_size_y: int = 32,
) -> torch.Tensor:
    K, X, group_size = _validate_awq_dequantize_contract(
        qweight,
        scales,
        zeros,
        block_size_x,
        block_size_y,
    )

    result = torch.empty(
        K,
        X * 8,
        device=qweight.device,
        dtype=scales.dtype,
    )

    Y = K

    grid = lambda META: (
        triton.cdiv(X, META["BLOCK_SIZE_X"]),
        triton.cdiv(Y, META["BLOCK_SIZE_Y"]),
    )
    awq_dequantize_kernel[grid](
        qweight,
        scales,
        zeros,
        group_size,
        result,
        X,
        Y,
        BLOCK_SIZE_X=block_size_x,
        BLOCK_SIZE_Y=block_size_y,
    )

    return result
