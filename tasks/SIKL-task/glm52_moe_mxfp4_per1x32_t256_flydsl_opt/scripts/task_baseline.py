# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Performance baseline for the GLM-5.2 MXFP4 MoE workload.

Extracted from the ``baseline`` field of the workload-schema definition
``aiter_fused_moe_per_1x32_d6144_e257_topk9_n512_k3072_i128``. Two edits the
schema field needs to run standalone: the omitted ``Optional`` import, and
``import aiter.fused_moe`` instead of ``import aiter`` (the latter raises
``AttributeError: module 'aiter' has no attribute 'fused_moe'``).

This is the production implementation the ported FlyDSL kernel is scored
against: ``aiter.fused_moe.fused_moe`` at ``QuantType.per_1x32``. On gfx950 it
dispatches to the FlyDSL two-stage grouped GEMMs plus HIP quant/sort/reduce
kernels; see ``forge_driver.py`` for the resolved implementation chain.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


def _fused_moe_baseline(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    activation: int = 0,
    doweight_stage1: bool = False,
) -> torch.Tensor:
    """Perf ground truth: aiter's own real fused_moe kernel, not the reference above."""
    # The schema field says `import aiter`, which leaves `aiter.fused_moe`
    # unbound unless another module already imported the submodule.
    import aiter.fused_moe  # noqa: F401
    from aiter import ActivationType, QuantType

    quant_type = QuantType.No if w1_scale is None else QuantType.per_1x32
    return aiter.fused_moe.fused_moe(
        hidden_states,
        w1,
        w2,
        topk_weight,
        topk_ids,
        activation=ActivationType(activation),
        quant_type=quant_type,
        doweight_stage1=doweight_stage1,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
    )


# ----- entry point -----
def run(*args, **kwargs):
    return _fused_moe_baseline(*args, **kwargs)
