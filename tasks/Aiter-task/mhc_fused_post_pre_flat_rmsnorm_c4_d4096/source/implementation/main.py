from __future__ import annotations
import torch

# ----- baseline -----
# Generated from @sikl_proxy: the annotated entry point is the ground truth.
def run(x, residual, post_mix, comb_mix, proj_weight, mix_scale, mix_bias, rms_eps, pre_eps, sinkhorn_eps, post_multiplier, sinkhorn_iters, norm_weight, norm_eps):
    from aiter.ops.mhc import mhc_fused_post_pre
    return mhc_fused_post_pre(layer_input=x, residual_in=residual, post_layer_mix=post_mix, comb_res_mix=comb_mix, fn=proj_weight, hc_scale=mix_scale, hc_base=mix_bias, rms_eps=rms_eps, hc_pre_eps=pre_eps, hc_sinkhorn_eps=sinkhorn_eps, hc_post_mult_value=post_multiplier, sinkhorn_repeat=sinkhorn_iters, norm_weight=norm_weight, norm_eps=norm_eps)
