# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""FlyDSL port of the GLM-5.2 MXFP4 MoE operator -- starter stub.

``forge-rewrite-by-flydsl`` replaces this file with a FlyDSL implementation.

The module must expose one factory, named by ``builder_symbol`` in the task's
workload.json (the same value KernelForge passes to the driver as
``KERNELFORGE_REWRITE_BUILDER_SYMBOL``, and the only name the harness looks up)::

    build_<operator>_module(num_tokens, model_dim, inter_dim, num_experts, topk)
        -> launch

    launch(hidden_states, w1, w2, topk_weight, topk_ids,
           w1_scale, w2_scale, activation, doweight_stage1)
        -> out                          # bf16 [num_tokens, model_dim]

where model_dim is the hidden size (6144), inter_dim the per-expert intermediate
size (256, with w1 holding [gate | up]), num_experts the local expert count (257)
and topk the experts per token (9).

The name is per-shape, so this stub deliberately defines no factory rather than
hardcoding one shape's symbol: ``scripts/forge_driver.py`` owns the operator
definition, the tensor layouts and where the baseline implementation lives, and
the factory is written here by the port session.

While the factory is absent the harness scores the operator's own baseline
(``aiter.fused_moe``), which is what Arena measures before the agent runs.
"""

from __future__ import annotations
