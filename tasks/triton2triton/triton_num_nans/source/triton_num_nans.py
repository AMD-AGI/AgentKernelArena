"""Count NaN values in logits kernel from vLLM v1 worker/gpu/metrics/logits.py."""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _num_nans_kernel(
    logits_ptr,
    logits_stride,
    num_nans_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    num_nans = 0
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < vocab_size
        logits = tl.load(
            logits_ptr + req_idx * logits_stride + block, mask=mask, other=0
        )
        logits = logits.to(tl.float32)
        is_nan = libdevice.isnan(logits).to(tl.int1)
        num_nans += tl.sum(is_nan).to(tl.int32)
    tl.store(num_nans_ptr + req_idx, num_nans)


def get_num_nans(logits: torch.Tensor) -> torch.Tensor:
    """Count the number of NaN values per row in a 2D logits tensor.

    Args:
        logits: [num_reqs, vocab_size] tensor
    Returns:
        num_nans: [num_reqs] int32 tensor with NaN counts
    """
    num_reqs, vocab_size = logits.shape
    BLOCK_SIZE = 8192
    num_nans = torch.empty(num_reqs, dtype=torch.int32, device=logits.device)
    _num_nans_kernel[(num_reqs,)](
        logits,
        logits.stride(0),
        num_nans,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return num_nans
