# SPDX-License-Identifier: MIT
"""Starting program: the complete pinned upstream AITER decode call."""


def fused_recurrent_kda(**kwargs):
    from aiter.ops.triton.attention.kda import fused_recurrent_kda as upstream

    return upstream(**kwargs)
