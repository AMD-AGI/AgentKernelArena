# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Correctness reference for this workload, copied from the schema bundle.

Everything below this docstring is the definition's ``reference`` callback, taken
without edit from schema v2 so that this task and the acceptance run that
verifies its result apply one implementation rather than two that agree today.
Do not edit it here: change it in the bundle and copy it down again, or the two
silently diverge -- which is exactly the failure this file exists to prevent.

``run`` is the entry point the bundle exports.
"""

from __future__ import annotations
import torch as torch

def _mhc_post_reference(x, residual, post_mix, comb_mix):
    """Combine old streams into new streams, then add the weighted layer output."""
    post = post_mix.reshape(residual.shape[0], residual.shape[1], 1)
    combined = torch.bmm(comb_mix.float().mT, residual.float())
    return (x.float().unsqueeze(1) * post.float() + combined).to(residual.dtype)


def _mhc_pre_reference(
    residual,
    proj_weight,
    mix_scale,
    mix_bias,
    rms_eps=1e-6,
    pre_eps=1e-6,
    sinkhorn_eps=1e-6,
    post_multiplier=1.0,
    sinkhorn_iters=20,
    norm_weight=None,
    norm_eps=1e-6,
):
    """Project residuals, apply RMS scaling, derive mixes, and aggregate streams."""
    streams = residual.shape[1]
    flat = residual.flatten(1).float()
    mixes = (flat @ proj_weight.float().T) * torch.rsqrt(
        flat.square().mean(dim=-1, keepdim=True) + rms_eps
    )
    pre_logits = mixes[:, :streams] * mix_scale[0] + mix_bias[:streams]
    post_logits = (
        mixes[:, streams : 2 * streams] * mix_scale[1] + mix_bias[streams : 2 * streams]
    )
    comb_logits = mixes[:, 2 * streams :] * mix_scale[2] + mix_bias[2 * streams :]
    pre_mix = torch.sigmoid(pre_logits).unsqueeze(-1) + pre_eps
    post_mix = (torch.sigmoid(post_logits) * post_multiplier).unsqueeze(-1)
    comb_mix = comb_logits.reshape(-1, streams, streams).softmax(dim=-1)
    comb_mix = comb_mix + sinkhorn_eps
    comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + sinkhorn_eps)
    for _ in range(sinkhorn_iters - 1):
        comb_mix = comb_mix / (comb_mix.sum(dim=-1, keepdim=True) + sinkhorn_eps)
        comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + sinkhorn_eps)
    layer_input = (residual.float() * pre_mix).sum(dim=1)
    if norm_weight is not None:
        layer_input = (
            layer_input
            * torch.rsqrt(layer_input.square().mean(dim=-1, keepdim=True) + norm_eps)
            * norm_weight.float()
        )
    return post_mix, comb_mix, layer_input.to(residual.dtype)


def _mhc_fused_post_pre_reference(
    x,
    residual,
    post_mix,
    comb_mix,
    proj_weight,
    mix_scale,
    mix_bias,
    rms_eps=1e-6,
    pre_eps=1e-6,
    sinkhorn_eps=1e-6,
    post_multiplier=1.0,
    sinkhorn_iters=20,
    norm_weight=None,
    norm_eps=1e-6,
):
    """Compose post→pre, preserving the BF16 residual boundary and AITER order."""
    next_residual = _mhc_post_reference(x, residual, post_mix, comb_mix)
    next_post_mix, next_comb_mix, layer_input = _mhc_pre_reference(
        next_residual,
        proj_weight,
        mix_scale,
        mix_bias,
        rms_eps,
        pre_eps,
        sinkhorn_eps,
        post_multiplier,
        sinkhorn_iters,
        norm_weight,
        norm_eps,
    )
    return next_post_mix, next_comb_mix, layer_input, next_residual


_callable = _mhc_fused_post_pre_reference


def run(*args, **kwargs):
    return _callable(*args, **kwargs)
