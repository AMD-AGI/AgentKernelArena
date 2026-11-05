# SPDX-License-Identifier: MIT
"""Complete upstream AITER MoE starting point; the protected runner owns data."""


def fused_moe(hidden_states, w1, w2, topk_weight, topk_ids, **kwargs):
    from aiter.fused_moe import fused_moe as upstream_fused_moe

    return upstream_fused_moe(hidden_states, w1, w2, topk_weight, topk_ids, **kwargs)
