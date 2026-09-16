# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Performance baseline for the MXFP4 fused-MoE workload.

Transcribed from the workload schema's ``baseline`` solution for this operator
(``solutions/baseline/moe/*/fused_moe_aiter_baseline_*.json``, entry point
``main.py::run``). This is the production implementation the ported FlyDSL
kernel is scored against: ``aiter.fused_moe.fused_moe`` at
``QuantType.per_1x32``.

Two documented divergences from the schema source, so a future comparison
against the bundle knows what to expect:

1. ``from typing import Optional`` is added. The schema source annotates the
   scale arguments with it and never imports it; that survives there only
   because ``from __future__ import annotations`` keeps annotations
   unevaluated.
2. The schema source's unused ``math`` and ``torch.nn.functional`` imports are
   dropped.
"""

from __future__ import annotations

from typing import Optional

import torch


def _fused_moe_baseline(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    activation: int = 0,
    doweight_stage1: bool = False,
) -> torch.Tensor:
    """Perf ground truth: aiter's own real fused_moe kernel, not the reference above."""
    import aiter.fused_moe
    from aiter import ActivationType, QuantType

    # aiter picks its preshuffled path off a python attribute, and no tensor
    # serialization carries one. This definition declares w1/w2 preshuffled --
    # that is the layout sglang always hands the kernel, and the reference
    # un-shuffles them on the way in -- so restore the tag the workload cannot
    # store. Without it the EP shapes compute a different answer entirely, and
    # every shape runs slower.
    w1.is_shuffled = True
    w2.is_shuffled = True
    quant_type = QuantType.No if w1_scale is None else QuantType.per_1x32
    return aiter.fused_moe.fused_moe(
        hidden_states,
        w1,
        w2,
        topk_weights,
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
