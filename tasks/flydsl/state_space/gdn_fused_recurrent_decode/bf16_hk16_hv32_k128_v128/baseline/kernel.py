# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Gated-delta-rule decode step with AITER's FlyDSL GDR decode kernel and its tuned launch table.

`run` is the launch of aiter.ops.flydsl.linear_attention_kernels.flydsl_gdr_decode for one token per
sequence, bfloat16 activations, float32 state in the [pool, heads, V, K] layout and in-kernel q/k L2
normalization (`use_qk_l2norm=True`, `need_shuffle_state=False`), with the launch configuration looked up as
`get_default_kwargs` does.
"""
import flydsl.expr as fx
import torch

from .gdr_decode import create_vk_gdr_decode_kernel
from .tensor_shim import _run_compiled, get_dtype_str
from .tuned_configs import DEFAULT_CONFIG, TUNED_CONFIGS


def run(*, query, key, value, a, b, dt_bias, A_log, state_indices, state, out):
    batch_size, seq_length, num_k_heads, head_k_dim = query.shape
    num_v_heads, head_v_dim = value.shape[-2], value.shape[-1]
    blocks_per_v, num_warps, warp_threads_k = TUNED_CONFIGS.get(batch_size, DEFAULT_CONFIG)
    exe = create_vk_gdr_decode_kernel(
        get_dtype_str(query.dtype),
        get_dtype_str(A_log.dtype),
        get_dtype_str(state.dtype),
        seq_length,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        state.stride(),
        a.stride(),
        b.stride(),
        True,
        NUM_BLOCKS_PER_V_DIM=blocks_per_v,
        NUM_WARPS=num_warps,
        WARP_THREADS_K=warp_threads_k,
    )
    _run_compiled(exe, query, key, value, a, b, dt_bias, A_log, state_indices, state, out, batch_size,
                  fx.Stream(torch.cuda.current_stream(query.device)))
