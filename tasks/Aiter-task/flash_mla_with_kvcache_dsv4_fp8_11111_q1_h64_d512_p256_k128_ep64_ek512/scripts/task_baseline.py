# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Performance baseline for this workload, copied from the schema bundle.

Everything below this docstring is the bundle's ``baseline`` solution, taken
without edit from schema v2. It is the production implementation a ported FlyDSL
kernel is scored against, so the task measures the same thing the acceptance run
measures. Do not edit it here: change it in the bundle and copy it down again.

``run`` is the entry point the bundle exports.
"""

from __future__ import annotations
from functools import partial as partial

def _baseline(*args, names, return_lse, **kwargs):
    from sglang.kernels.ops.attention.dsa.tilelang_kernel import (
        dpsk_v4_fp8_attention_fwd,
    )

    if len(args) > len(names):
        raise TypeError("too many MLA baseline inputs")
    call = dict(zip(names, args))
    call.update(kwargs)
    output, lse = dpsk_v4_fp8_attention_fwd(
        q=call["q"],
        k_cache=call["kv_cache"],
        block_table=None,
        cache_seqlens=None,
        head_dim_v=call["q"].shape[-1],
        tile_scheduler_metadata=None,
        indices=call["sparse_indices"],
        topk_length=call.get("sparse_lens"),
        extra_k_cache=call.get("extra_kv_cache"),
        extra_indices_in_kvcache=call.get("extra_sparse_indices"),
        extra_topk_length=call.get("extra_sparse_lens"),
        softmax_scale=call["sm_scale"],
        attn_sink=call.get("sinks"),
    )
    return (output, lse) if return_lse else output


_callable = partial(_baseline, **{'names': ('q', 'kv_cache', 'sparse_indices', 'sparse_lens', 'extra_kv_cache', 'extra_sparse_indices', 'extra_sparse_lens', 'sm_scale', 'sinks',), 'return_lse': True})


def run(*args, **kwargs):
    return _callable(*args, **kwargs)
