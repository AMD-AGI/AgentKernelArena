"""Compute identity kernel using Triton, adapted from vLLM fused_moe.py.

Copies hidden_states scaled by expert_scales for MoE identity/zero experts.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def compute_identity_kernel(
    top_k: int,
    hidden_states_ptr: tl.tensor,
    expert_scales_ptr: tl.tensor,
    num_tokens: int,
    output_ptr: tl.tensor,
    hidden_dim: int,
    scales_stride: int,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """
    For each token, compute output = sum_i(hidden_states * expert_scales[i])
    across top_k expert scale values. This is used for "identity" zero-experts
    in MoE models.
    """
    pid = tl.program_id(0)

    batch_id = pid // (hidden_dim // BLOCK_SIZE)
    dim_offset = pid % (hidden_dim // BLOCK_SIZE) * BLOCK_SIZE

    if batch_id >= num_tokens or dim_offset >= hidden_dim:
        return

    h = tl.load(
        hidden_states_ptr
        + batch_id * hidden_dim
        + dim_offset
        + tl.arange(0, BLOCK_SIZE),
        mask=(dim_offset + tl.arange(0, BLOCK_SIZE)) < hidden_dim,
    )

    result = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(top_k):
        scale = tl.load(expert_scales_ptr + batch_id * scales_stride + i)
        result += h * scale

    tl.store(
        output_ptr + batch_id * hidden_dim + dim_offset + tl.arange(0, BLOCK_SIZE),
        result,
        mask=(dim_offset + tl.arange(0, BLOCK_SIZE)) < hidden_dim,
    )


def compute_identity(
    hidden_states: torch.Tensor,
    expert_scales: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """
    Compute identity scaling: output[i] = sum_k(hidden_states[i] * expert_scales[i, k]).

    Args:
        hidden_states: [num_tokens, hidden_dim]
        expert_scales: [num_tokens, top_k]
        top_k: number of expert scale values per token
    Returns:
        output: [num_tokens, hidden_dim]
    """
    assert hidden_states.dim() == 2
    assert expert_scales.dim() == 2
    num_tokens, hidden_dim = hidden_states.shape

    output = torch.zeros_like(hidden_states)
    BLOCK_SIZE = 256

    grid = (num_tokens * (hidden_dim // BLOCK_SIZE),)
    compute_identity_kernel[grid](
        top_k,
        hidden_states,
        expert_scales,
        num_tokens,
        output,
        hidden_dim,
        expert_scales.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output
