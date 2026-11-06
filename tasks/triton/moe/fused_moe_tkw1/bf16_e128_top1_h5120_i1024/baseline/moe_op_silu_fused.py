# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from typing import Any

import torch
import triton
import triton.language as tl

from .moe_op_silu_fused_kernels import _fused_moe_silu_kernel

# Source:
# MoE Kernel adapted from VLLM
# (ROCm/aiter aiter/ops/triton/moe/moe_op_silu_fused.py: fused_moe_silu reduced to its unquantized,
# non-persistent launch; the logger, the persistent/GPTQ-AWQ branches and the module setters are removed.)

_PADDING_SIZE = 0


def get_num_xcds():
    # Currently, you can't query this programmatically. For gfx942/gfx950 it's 8, so we hardcode that here.
    return 8


def fused_moe_silu(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: torch.Tensor | None,
    B_scale: torch.Tensor | None,
    B_zp: torch.Tensor | None,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    compute_type: tl.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    block_shape: list[int] | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    """
    Fused MoE computation with SiLU activation and optional quantization.

    Args:
        A (torch.Tensor): Input activations with shape (num_tokens, hidden_dim).
        B (torch.Tensor): Expert weights with shape (num_experts, hidden_dim, intermediate_dim).
        C (torch.Tensor): Output tensor with shape (num_tokens, top_k, intermediate_dim).
        topk_weights (torch.Tensor): Routing weights for top-k experts with shape (num_tokens, top_k).
        topk_ids (torch.Tensor): Top-k expert IDs per token with shape (num_tokens, top_k).
        sorted_token_ids (torch.Tensor): Token IDs sorted by expert assignment.
        expert_ids (torch.Tensor): Expert ID for each sorted token.
        num_tokens_post_padded (torch.Tensor): Total tokens after block-size padding with shape (1,).
        mul_routed_weight (bool): Multiply output by routing weights.
        top_k (int): Number of experts per token.
        compute_type (tl.dtype): Computation dtype for accumulation.
        config (Optional[Dict[str, Any]]): Kernel tuning parameters (BLOCK_SIZE_M, BLOCK_SIZE_N,
            BLOCK_SIZE_K, GROUP_SIZE_M).

    Returns:
        None. Results written in-place to C with SiLU activation applied.
    """
    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1
    # Only the unquantized path is vendored.
    assert not (use_fp8_w8a8 or use_int8_w8a16 or use_int4_w4a16)
    assert A_scale is None
    assert B_scale is None

    EM = sorted_token_ids.shape[0]
    if A.shape[0] < config["BLOCK_SIZE_M"]:
        # optimize for small batch_size.
        # We assume that top_ids of each token is unique, so
        # so num_valid_experts <= batch_size <= BLOCK_SIZE_M,
        # and we can skip some invalid blocks.
        EM = min(sorted_token_ids.shape[0], A.shape[0] * top_k * config["BLOCK_SIZE_M"])

    grid = lambda META: (
        triton.cdiv(EM, META["BLOCK_SIZE_M"])
        * triton.cdiv(B.shape[1], META["BLOCK_SIZE_N"]),
    )
    _fused_moe_silu_kernel[grid](
        A,
        B,
        C,
        A_scale,
        B_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.shape[1],
        A.shape[1] - _PADDING_SIZE,
        topk_ids.numel(),
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        A_scale.stride(0) if A_scale is not None and A_scale.ndim == 2 else 0,
        A_scale.stride(1) if A_scale is not None and A_scale.ndim == 2 else 0,
        B_scale.stride(0) if B_scale is not None and B_scale.ndim >= 2 else 0,
        B_scale.stride(2) if B_scale is not None and B_scale.ndim == 3 else 0,
        B_scale.stride(1) if B_scale is not None and B_scale.ndim >= 2 else 0,
        0 if block_shape is None else block_shape[0],
        0 if block_shape is None else block_shape[1],
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        NUM_XCDS=get_num_xcds(),
        **config,
    )
