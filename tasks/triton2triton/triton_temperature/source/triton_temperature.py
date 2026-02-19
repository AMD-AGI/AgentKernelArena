"""Temperature scaling kernel, adapted from vLLM gumbel.py."""
import torch
import triton
import triton.language as tl


@triton.jit
def _temperature_kernel(
    logits_ptr,
    logits_stride,
    idx_mapping_ptr,
    temperature_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)
    temperature = tl.load(temperature_ptr + req_state_idx).to(tl.float32)
    if temperature == 0.0 or temperature == 1.0:
        return

    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size

    logits = tl.load(logits_ptr + batch_idx * logits_stride + block, mask=mask)
    logits = logits.to(tl.float32)
    logits = logits / temperature
    tl.store(logits_ptr + batch_idx * logits_stride + block, logits, mask=mask)


def apply_temperature(logits, idx_mapping, temperature):
    """Apply temperature scaling to logits in-place."""
    num_reqs, vocab_size = logits.shape
    BLOCK_SIZE = 8192
    num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)
    _temperature_kernel[(num_reqs, num_blocks)](
        logits, logits.stride(0), idx_mapping, temperature,
        vocab_size, BLOCK_SIZE=BLOCK_SIZE,
    )
