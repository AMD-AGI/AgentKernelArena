# SPDX-License-Identifier: MIT
"""Starting implementation: the complete pinned upstream AITER public call.

Replace this implementation and add helpers under source/ to reorganize the
complete operation. The protected runner owns inputs, oracle and timing.
"""


def fused_moe(hidden_states, w1, w2, topk_weight, topk_ids, **kwargs):
    from aiter.fused_moe import fused_moe as upstream_fused_moe

    return upstream_fused_moe(hidden_states, w1, w2, topk_weight, topk_ids, **kwargs)
