"""Batched memcpy kernel from vLLM Mamba worker utils.

Generic batched memory copy using void pointer casting. Each program instance
copies one (src, dst, size) triple using byte-level access.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def batch_memcpy_kernel(src_ptrs, dst_ptrs, sizes, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)

    src_ptr = tl.load(src_ptrs + pid)
    dst_ptr = tl.load(dst_ptrs + pid)
    size = tl.load(sizes + pid)

    offsets = tl.arange(0, BLOCK_SIZE)
    for i in range(0, size, BLOCK_SIZE):
        mask = (i + offsets) < size

        curr_src_ptr = (src_ptr + i + offsets).to(tl.pointer_type(tl.uint8))
        curr_dst_ptr = (dst_ptr + i + offsets).to(tl.pointer_type(tl.uint8))

        data = tl.load(curr_src_ptr, mask=mask)
        tl.store(curr_dst_ptr, data, mask=mask)


def batch_memcpy(src_ptrs: torch.Tensor, dst_ptrs: torch.Tensor, sizes: torch.Tensor) -> None:
    """
    Batched memory copy.

    Args:
        src_ptrs: [batch] int64 tensor of source memory addresses
        dst_ptrs: [batch] int64 tensor of destination memory addresses
        sizes: [batch] int32 tensor of copy sizes in bytes
    """
    batch = src_ptrs.shape[0]
    assert dst_ptrs.shape[0] == batch
    assert sizes.shape[0] == batch

    grid = (batch,)
    BLOCK_SIZE = 1024
    batch_memcpy_kernel[grid](src_ptrs, dst_ptrs, sizes, BLOCK_SIZE=BLOCK_SIZE)
