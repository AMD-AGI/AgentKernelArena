# SPDX-License-Identifier: Apache-2.0
# Vendored for AgentKernelArena from sgl-project/sgl-amd-mono at 52486703fbf27e77f05b573e4d931d4eec87118d,
# sgl_amd_mono/kimi_k3/__init__.py (the model constants the kernels build in), unchanged.
"""Kimi-K3 monokernels (RFC #43949): the spec and its SGLang glue in ``spec``,
and the gfx950 kernels and host runner in ``gfx950``.

Adapted from vLLM PR #60683, whose shared stages come from ROCm/ATOM
(https://github.com/ROCm/ATOM).
"""

# Kimi-K3's shapes and routing, which the kernels build in. These live on the
# checkpoint's text config: the top-level config of the VLM wrapper, which is
# what the architecture names, carries none of them.
HF_CONFIG = dict(
    hidden_size=7168,
    routed_expert_hidden_size=3584,
    num_experts=896,
    num_experts_per_token=16,
    num_shared_experts=2,
    moe_router_activation_func="sigmoid",
    routed_scaling_factor=1.0,
    first_k_dense_replace=1,
    moe_layer_freq=1,
)
# The KDA layers' shapes. The launch runs no attention, but the seam it starts
# from is sized by them.
LINEAR_ATTN_CONFIG = dict(head_dim=128, num_heads=96, short_conv_kernel_size=4)

# One launch's token rows at most: the x rows fit one CTA's LDS.
MAX_ROWS = 8
# The attention's TP-local output per rank at TP8: 12 heads of 128. o_proj's
# input, which is where the launch starts.
CORE = 1536
# Attention-residual bank rows readable at a layer's MLP seam.
MAX_BLOCKS = 8
