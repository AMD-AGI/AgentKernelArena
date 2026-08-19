# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Standalone sigmoid top-1 MoE routing Triton kernel.

Source: aiter/ops/triton/moe/moe_routing_sigmoid_top1_fused.py (+ _triton_kernels).
"""

from typing import Optional
import functools
import torch
import triton
import triton.language as tl


# --- inlined arch detection (utils._triton.arch_info) ---
try:
    _CACHED_ARCH = triton.runtime.driver.active.get_current_target().arch
except RuntimeError:
    _CACHED_ARCH = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).gcnArchName.split(":")[0]


def get_arch():
    return _CACHED_ARCH


# --- inlined constexpr-aware kernel naming (utils._triton.kernel_repr) ---
def _sanitize_constexpr_value(value):
    if value is None:
        return "NONE"
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return str(value)

    # for lists, tuples, sets - recursively join each
    if isinstance(value, (list, tuple, set)):
        items = sorted(value, key=str) if isinstance(value, set) else value
        sanitized_items = [_sanitize_constexpr_value(item) for item in items]
        joined = "_".join(sanitized_items)
        return joined if joined else "NONE"

    if isinstance(value, str):
        cleaned_value = "".join(ch if ch.isalnum() else "_" for ch in value).strip("_")
        return cleaned_value.upper() if cleaned_value else "NONE"

    cleaned_value = "".join(ch if ch.isalnum() else "_" for ch in str(value)).strip("_")
    return cleaned_value.upper() if cleaned_value else "NONE"


def make_kernel_repr(base_name, config_keys, name_key=None):
    # When name_key is set, the base name is taken from the matching constexpr
    # kwarg at call time (falling back to base_name if missing/empty). Lets a
    # single shared kernel produce caller-specific names in compiled artifacts.
    def _repr(specialization):
        constants = specialization.constants

        name = base_name
        if name_key is not None:
            override = constants.get(name_key, None)
            if override:
                cleaned = "".join(
                    ch if ch.isalnum() or ch == "_" else "_" for ch in str(override)
                )
                if cleaned:
                    name = cleaned

        name_parts = []
        for key in config_keys:
            value = constants.get(key, None)
            symbol = _sanitize_constexpr_value(value)
            name_parts.append(f"{key}_{symbol}")

        if not name_parts:
            return name

        suffix = "_".join(name_parts)
        return f"{name}_{suffix}"

    return _repr


# Tuning tables for configs/moe/{gfx950,gfx942}-MOE_ROUTING_SIGMOID_TOPK1.json
# (loaded from disk in the original _get_config).
_ROUTING_SIGMOID_TOPK1_CONFIGS = {
    "gfx950": {
        "N16": {
            "small": {"BLOCK_M": 16, "BLOCK_K": 256, "num_warps": 4, "num_stages": 2, "waves_per_eu": 3, "kpack": 1},
            "medium": {"BLOCK_M": 16, "BLOCK_K": 256, "num_warps": 4, "num_stages": 2, "waves_per_eu": 3, "kpack": 1},
            "large": {"BLOCK_M": 16, "BLOCK_K": 256, "num_warps": 4, "num_stages": 2, "waves_per_eu": 3, "kpack": 1},
            "xlarge": {"BLOCK_M": 32, "BLOCK_K": 128, "num_warps": 8, "num_stages": 2, "waves_per_eu": 2, "kpack": 1},
        },
        "N128": {
            "small": {"BLOCK_M": 16, "BLOCK_K": 256, "num_warps": 8, "num_stages": 1, "waves_per_eu": 0, "kpack": 1},
            "medium": {"BLOCK_M": 16, "BLOCK_K": 256, "num_warps": 8, "num_stages": 1, "waves_per_eu": 0, "kpack": 1},
            "large": {"BLOCK_M": 16, "BLOCK_K": 256, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "kpack": 1},
            "xlarge": {"BLOCK_M": 32, "BLOCK_K": 128, "num_warps": 8, "num_stages": 2, "waves_per_eu": 2, "kpack": 1},
        },
    },
    "gfx942": {
        "N16": {
            "small": {"BLOCK_M": 16, "BLOCK_K": 256, "num_warps": 4, "num_stages": 2, "waves_per_eu": 3, "kpack": 1},
            "medium": {"BLOCK_M": 16, "BLOCK_K": 256, "num_warps": 4, "num_stages": 2, "waves_per_eu": 3, "kpack": 1},
            "large": {"BLOCK_M": 16, "BLOCK_K": 256, "num_warps": 4, "num_stages": 2, "waves_per_eu": 3, "kpack": 2},
            "xlarge": {"BLOCK_M": 32, "BLOCK_K": 128, "num_warps": 8, "num_stages": 2, "waves_per_eu": 2, "kpack": 2},
        },
        "N128": {
            "small": {"BLOCK_M": 16, "BLOCK_K": 256, "num_warps": 8, "num_stages": 1, "waves_per_eu": 0, "kpack": 1},
            "medium": {"BLOCK_M": 16, "BLOCK_K": 256, "num_warps": 8, "num_stages": 1, "waves_per_eu": 0, "kpack": 2},
            "large": {"BLOCK_M": 16, "BLOCK_K": 256, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "kpack": 2},
            "xlarge": {"BLOCK_M": 32, "BLOCK_K": 128, "num_warps": 8, "num_stages": 2, "waves_per_eu": 2, "kpack": 2},
        },
    },
}


_routing_sigmoid_top1_repr = make_kernel_repr(
    "_routing_sigmoid_top1_kernel",
    [
        "BLOCK_M",
        "BLOCK_K",
        "BLOCK_N",
        "TOPK",
        "FUSED_SHARED_EXPERTS",
    ],
)


@triton.jit(repr=_routing_sigmoid_top1_repr)
def _routing_sigmoid_top1_kernel(
    X_ptr,
    W_ptr,
    topk_ids_ptr,
    topk_weights_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wk,
    stride_wn,
    stride_topk_ids_m,
    stride_topk_ids_n,
    stride_topk_weights_m,
    stride_topk_weights_n,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TOPK: tl.constexpr,
    FUSED_SHARED_EXPERTS: tl.constexpr,
):
    # Program ID corresponds to the block index in M dimension
    pid_m = tl.program_id(axis=0)

    # Offsets for the current block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    _TOPK: tl.constexpr = TOPK + 1 if FUSED_SHARED_EXPERTS else TOPK

    offs_topk = tl.arange(0, _TOPK)

    # Masks for bounds checking
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Initialize accumulator for matmul (will be in float32 due to default acc_type)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks of BLOCK_K
    for k in range(0, K, BLOCK_K):
        # Compute pointers for A and B
        offs_k_iter = k + offs_k
        mask_k = offs_k_iter < K

        X_ptrs = X_ptr + (
            # pyre-ignore
            offs_m[:, None] * stride_xm
            + offs_k_iter[None, :] * stride_xk
        )
        W_ptrs = W_ptr + (
            offs_k_iter[:, None] * stride_wk + offs_n[None, :] * stride_wn
        )

        # Load A and B tiles
        # pyre-ignore
        x = tl.load(X_ptrs, mask=(mask_m[:, None] & mask_k[None, :]), other=0.0)
        w = tl.load(W_ptrs, mask=(mask_k[:, None] & mask_n[None, :]), other=0.0)

        # Compute partial matmul for the current block using FP16 inputs and FP32 accumulation
        acc = tl.dot(x, w, acc=acc)

    acc = tl.sigmoid(acc)
    # Get topk results
    topk_ids = tl.argmax(acc, axis=1, tie_break_left=True)  # Shape: (BLOCK_M,)
    topk_weights = tl.max(acc, axis=1)  # Shape: (BLOCK_M,)

    # Create buffers for results
    topk_ids_buffer = tl.zeros((BLOCK_M, _TOPK), dtype=tl.int32)
    topk_weights_buffer = tl.zeros((BLOCK_M, _TOPK), dtype=tl.float32)

    if FUSED_SHARED_EXPERTS:
        # Set the first column with broadcasting
        topk_ids_buffer = tl.where(
            (offs_topk[None, :] < _TOPK - 1), topk_ids[:, None], N
        )
        topk_weights_buffer = tl.where(
            (offs_topk[None, :] < _TOPK - 1), topk_weights[:, None], 1.0
        )
    else:
        topk_ids_buffer = topk_ids[:, None]
        topk_weights_buffer = topk_weights[:, None]

    topk_ids_ptrs = (
        topk_ids_ptr
        + offs_m[:, None] * stride_topk_ids_m
        + offs_topk[None, :] * stride_topk_ids_n
    )

    topk_weights_ptrs = (
        topk_weights_ptr
        + offs_m[:, None] * stride_topk_weights_m
        + offs_topk[None, :] * stride_topk_weights_n
    )

    tl.store(topk_ids_ptrs, topk_ids_buffer)
    tl.store(topk_weights_ptrs, topk_weights_buffer)


@functools.lru_cache(maxsize=1024)
def _get_config(M, N, K):
    config = _ROUTING_SIGMOID_TOPK1_CONFIGS[get_arch()]

    n_key = "N16" if N <= 16 else "N128"
    m_key = (
        "xlarge"
        if M >= 8192
        else "large" if M >= 4096 else "medium" if M >= 2048 else "small"
    )
    return config[n_key][m_key]


def routing_sigmoid_top1(
    x, w, topk, fused_shared_experts=False, config: Optional[dict[str, any]] = None
):
    """
    Computes top-1 MoE routing with sigmoid activation for expert selection.

    Args:
        x (torch.Tensor): Input activations with shape (batch_size, seq_len, hidden_dim) or (M, K).
        w (torch.Tensor): Routing weights with shape (hidden_dim, num_experts).
        topk (int): Number of experts to select. Must be 1.
        fused_shared_experts (bool): Include shared expert (always selected) alongside top-1.
        config (Optional[dict]): Kernel tuning parameters (BLOCK_M, BLOCK_K).

    Returns:
        tuple: (topk_ids, topk_weights)
            - topk_ids (torch.Tensor): Selected expert IDs with shape (M, topk) or (M, topk+1) if fused_shared_experts.
            - topk_weights (torch.Tensor): Routing weights (sigmoid scores) with shape (M, topk) or (M, topk+1).
    """
    x = x.view(-1, x.shape[-1])

    assert topk == 1

    # M: batch_size x seq_len, K: hidden_dim, N: num_experts
    M, K = x.shape
    Kb, N = w.shape
    assert K == Kb

    _topk = topk
    if fused_shared_experts:
        _topk += 1

    # Output tensor
    topk_ids = torch.empty((M, _topk), device=x.device, dtype=torch.int32)
    topk_weights = torch.empty((M, _topk), device=x.device, dtype=torch.float32)

    config = _get_config(M, N, K)

    # Grid size
    def grid(META):
        return (triton.cdiv(M, META["BLOCK_M"]),)

    _routing_sigmoid_top1_kernel[grid](
        x,
        w,
        topk_ids,
        topk_weights,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        topk_ids.stride(0),
        topk_ids.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        BLOCK_N=N,  # Set BLOCK_N to N
        TOPK=topk,
        FUSED_SHARED_EXPERTS=fused_shared_experts,
        **config,
    )

    return topk_ids, topk_weights
