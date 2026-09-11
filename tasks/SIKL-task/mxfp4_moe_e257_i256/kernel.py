# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""FlyDSL port of the MXFP4 fused-MoE operator -- starter stub.

``forge-rewrite-by-flydsl`` replaces this file with a FlyDSL implementation.

The module must expose one factory, named by ``builder_symbol`` in the task's
workload.json (the same value KernelForge passes to the driver as
``KERNELFORGE_REWRITE_BUILDER_SYMBOL``, and the only name the harness looks up)::

    build_<operator>_module(num_tokens, model_dim, inter_dim, num_experts, topk)
        -> launch

    launch(hidden_states, w1, w2, topk_weight, topk_ids,
           w1_scale, w2_scale, activation, doweight_stage1)
        -> out                          # bf16 [num_tokens, model_dim]

where w1 holds [gate | up] along its rows and the builder is called once per
scored case so all shape-dependent work stays outside the timed region.

The name is per-operator, so this stub deliberately defines no factory rather
than hardcoding one: ``scripts/forge_driver.py`` owns the operator definition,
the tensor layouts and where the baseline implementation lives, and the factory
is written here by the port session.

While the factory is absent the harness scores the operator's own baseline
(``aiter.fused_moe.fused_moe``), which is what Arena measures before the agent
runs.
"""

from __future__ import annotations
