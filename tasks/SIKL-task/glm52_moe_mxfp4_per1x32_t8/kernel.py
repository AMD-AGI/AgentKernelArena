# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""FlyDSL port of the GLM-5.2 MXFP4 MoE operator -- starter stub.

``forge-rewrite-by-flydsl`` replaces this file with a FlyDSL implementation.
The builder and launch signatures are defined by the measurement driver
(``scripts/forge_driver.py``), which also documents the operator, the tensor
layouts, and where the baseline implementation lives.

While this stub is in place the harness scores the operator's own baseline
(``aiter.fused_moe``), which is what Arena measures before the agent runs.
"""

from __future__ import annotations


def build_glm52_mxfp4_moe_2stage_module(
    num_tokens: int,
    model_dim: int,
    inter_dim: int,
    num_experts: int,
    topk: int,
):
    """Return a launch callable for the fused MXFP4 MoE layer.

    Args:
        num_tokens: Rows of the activation for this call.
        model_dim: Hidden size (6144).
        inter_dim: Per-expert intermediate size (256); w1 holds [gate | up].
        num_experts: Local experts on this rank (257).
        topk: Experts per token (9).

    Returns:
        ``launch(hidden_states, w1, w2, topk_weight, topk_ids, w1_scale,
        w2_scale, activation, doweight_stage1) -> bf16 [num_tokens, model_dim]``
    """
    raise NotImplementedError(
        "FlyDSL port not implemented yet; see scripts/forge_driver.py for the "
        "operator definition, the tensor layouts, and the required interface."
    )
