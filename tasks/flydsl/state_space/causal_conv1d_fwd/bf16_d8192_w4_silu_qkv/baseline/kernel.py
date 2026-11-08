# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Prefill causal conv1d with fused split q/k/v output: AITER's FlyDSL kernel and its launch.

`run` is the launch of aiter.ops.flydsl.causal_conv1d_flydsl.causal_conv1d_split_qkv_flydsl_fn with the
precomputed chunk schedule (the `metadata` path of that function: the program-to-(sequence, chunk) map comes
from the attention metadata, which the serving stack builds on the host), bias, SiLU and bfloat16, writing into
caller-provided q, k and v.
"""
import flydsl.compiler as flyc
import torch

from .causal_conv1d_flydsl import _get_compiled


def run(*, x, weight, bias, conv_states, query_start_loc, cache_indices, has_initial_state, chunk_sequence_ids,
        chunk_ids, query, key, value):
    dim, _ = x.shape
    _, width = weight.shape
    launcher = _get_compiled(int(width), True, True, 64, 64, 256, "bf16")
    tot = chunk_sequence_ids.shape[0]
    grid_y_dim = (dim + launcher._tn - 1) // launcher._tn
    # The conv state is [cache line, state position, channel]; the kernel takes [cache line, channel, position].
    states = conv_states.transpose(1, 2)

    launch_args = (
        x,
        weight,
        bias,
        states,
        cache_indices,
        has_initial_state,
        query_start_loc,
        chunk_sequence_ids,
        chunk_ids,
        query,
        key,
        value,
        int(dim),
        int(key.shape[1]),
        int(value.shape[1]),
        int(x.stride(0)),
        int(x.stride(1)),
        int(weight.stride(0)),
        int(weight.stride(1)),
        int(states.stride(0)),
        int(states.stride(1)),
        int(states.stride(2)),
        int(cache_indices.stride(0)),
        int(query.stride(0)),
        int(query.stride(1)),
        int(key.stride(0)),
        int(key.stride(1)),
        int(value.stride(0)),
        int(value.stride(1)),
        int(tot),
        int(grid_y_dim),
        torch.cuda.current_stream(x.device),
    )

    # First call compiles and executes in one step; later calls reuse the cached CompiledFunction.
    compiled = getattr(launcher, "_fast_compiled", None)
    if compiled is None:
        try:
            launcher._fast_compiled = flyc.compile(launcher, *launch_args)
        except Exception:  # noqa: BLE001
            launcher._fast_compiled = False  # fall back permanently
            launcher(*launch_args)
    elif compiled is not False:
        compiled(*launch_args)
    else:
        launcher(*launch_args)
