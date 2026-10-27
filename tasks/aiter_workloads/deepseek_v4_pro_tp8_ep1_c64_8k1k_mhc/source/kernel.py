# SPDX-License-Identifier: MIT
"""Starting implementation: the complete pinned upstream AITER public call.

Replace this implementation and add helpers under source/ to reorganize the
complete operation. The protected runner owns inputs, oracle and timing.
"""


def mhc_fused_post_pre(layer_input, residual_in, post_layer_mix, comb_res_mix,
                       fn, hc_scale, hc_base, **kwargs):
    from aiter.ops.mhc import mhc_fused_post_pre as upstream_mhc_fused_post_pre

    return upstream_mhc_fused_post_pre(
        layer_input, residual_in, post_layer_mix, comb_res_mix,
        fn, hc_scale, hc_base, **kwargs,
    )
