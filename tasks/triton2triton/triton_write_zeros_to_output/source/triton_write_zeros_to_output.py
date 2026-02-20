"""Write zeros to output kernel using Triton, adapted from vLLM fused_moe.py."""
import torch
import triton
import triton.language as tl


@triton.jit
def write_zeros_to_output(
    c_ptr,
    stride_cm,
    stride_cn,
    M,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Write zeros to a 2D output tensor in blocks.
    Each program handles one (block_m, block_n) tile.
    """
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    mask_m = offs_m < M
    mask_n = offs_n < N

    zeros = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float16)
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, zeros, mask=c_mask)


def write_zeros(output: torch.Tensor) -> torch.Tensor:
    """Zero out a 2D tensor using the Triton kernel."""
    assert output.dim() == 2, "Output must be 2D"
    M, N = output.shape
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
    write_zeros_to_output[grid](
        output,
        output.stride(0),
        output.stride(1),
        M,
        N,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    return output
