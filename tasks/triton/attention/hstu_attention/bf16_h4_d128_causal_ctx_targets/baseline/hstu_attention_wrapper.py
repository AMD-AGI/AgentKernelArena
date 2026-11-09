# Copyright (C) Advanced Micro Devices, Inc. All rights reserved.
# Copyright (C) 2024-2026, The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch

# @manual=//triton:triton
import triton

from .hstu_attention_kernels import _get_fwd_config, _hstu_attn_fwd


def prev_power_of_2(x: int) -> int:
    # aiter/ops/triton/utils/common_utils.py
    out = triton.next_power_of_2(x)
    return out // 2 if out > x else out


def triton_hstu_attention_fwd(
    N: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    causal: bool,
    num_targets: torch.Tensor | None,
    max_attn_len: int,
    contextual_seq_len: int,
    sort_by_length_indices: torch.Tensor | None,
    config: dict | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    HSTU attention forward pass with SiLU activation: Y = silu(alpha * (Q @ K^T)) @ V.
    Works with jagged tensors (variable-length sequences concatenated along batch dimension).

    Args:
        N (int): Maximum sequence length across all sequences.
        alpha (float): Scale factor applied to Q @ K^T before SiLU activation.
        q (torch.Tensor): Query jagged tensor with shape (total_tokens, num_heads, head_dim).
        k (torch.Tensor): Key jagged tensor with shape (total_tokens, num_heads, head_dim).
        v (torch.Tensor): Value jagged tensor with shape (total_tokens, num_heads, head_dim).
        seq_offsets (torch.Tensor): Sequence boundaries with shape (batch_size + 1,).
            Element i contains cumulative token count up to sequence i.
        causal (bool): Apply causal masking.
        num_targets (Optional[torch.Tensor]): Number of target tokens per sequence for masking.
        max_attn_len (int): Maximum attention span limit. 0 disables limit.
        contextual_seq_len (int): Contextual prefix length. 0 disables contextual masking.
        sort_by_length_indices (Optional[torch.Tensor]): Indices to process sequences in descending length order.
        config (Optional[dict]): Kernel tuning parameters (BLOCK_M, BLOCK_N).

    Returns:
        torch.Tensor: Output jagged tensor with shape (total_tokens, num_heads, head_dim).
    """
    Z = seq_offsets.numel() - 1
    AUTOTUNE_Z = prev_power_of_2(Z)
    L, H, DimQ = q.shape
    _, _, DimV = v.shape
    if out is None:
        out = torch.empty_like(v)
    has_multiple_targets = num_targets is not None
    has_contextual_seq_len = contextual_seq_len > 0
    has_max_attn_len = max_attn_len > 0
    has_sort_by_length_indices = sort_by_length_indices is not None
    if L == 0:
        return out

    DeltaSize = 0
    IS_DELTA_Q = False

    if config is None:
        config = _get_fwd_config(
            AUTOTUNE_Z,
        )

    grid = lambda meta: (
        triton.cdiv(N, meta["BLOCK_M"]),
        Z * H,
    )

    _hstu_attn_fwd[grid](
        Q=q,
        K=k,
        V=v,
        sort_by_length_indices=sort_by_length_indices,
        seq_offsets=seq_offsets,
        num_targets=num_targets,
        Out=out,
        stride_qm=q.stride(0),
        stride_qh=q.stride(1),
        stride_kn=k.stride(0),
        stride_kh=k.stride(1),
        stride_vn=v.stride(0),
        stride_vh=v.stride(1),
        stride_om=out.stride(0),
        stride_oh=out.stride(1),
        alpha=alpha,
        H=H,
        MAX_SEQ_LEN=N,
        DeltaSize=DeltaSize,
        contextual_seq_len=contextual_seq_len,
        max_attn_len=max_attn_len,
        CAUSAL=causal,
        HAS_MULTIPLE_TARGETS=has_multiple_targets,
        IS_DELTA_Q=IS_DELTA_Q,
        ALLOW_TF32=False,
        BLOCK_D_Q=DimQ,
        BLOCK_D_V=DimV,
        HAS_CONTEXTUAL_SEQ_LEN=has_contextual_seq_len,
        HAS_MAX_ATTN_LEN=has_max_attn_len,
        HAS_SORT_BY_LENGTH_INDICES=has_sort_by_length_indices,
        **config,
    )

    return out
