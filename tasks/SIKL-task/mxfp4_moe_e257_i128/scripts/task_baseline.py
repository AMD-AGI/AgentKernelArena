# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Performance baseline for this workload, copied from the schema bundle.

Everything below this docstring is the bundle's ``baseline`` solution, taken
without edit from schema v2. It is the production implementation a ported FlyDSL
kernel is scored against, so the task measures the same thing the acceptance run
measures. Do not edit it here: change it in the bundle and copy it down again.

``run`` is the entry point the bundle exports.
"""

from __future__ import annotations
from typing import Optional as Optional
import torch as torch

def _fused_moe_baseline(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: int = 0,
    doweight_stage1: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Perf ground truth: aiter's own real fused_moe kernel, not the reference above."""
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe

    # Restore the preshuffle tag lost during tensor serialization.
    w1.is_shuffled = True  # type: ignore[attr-defined]
    w2.is_shuffled = True  # type: ignore[attr-defined]
    # AITER initializes its enums dynamically in the GPU runtime.
    quant_type = QuantType.No if w1_scale is None else QuantType.per_1x32  # type: ignore[union-attr]
    return fused_moe(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation=ActivationType(0 if activation is None else activation),  # type: ignore[call-arg]
        quant_type=quant_type,
        doweight_stage1=doweight_stage1,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
    )


_callable = _fused_moe_baseline


def run(*args, **kwargs):
    return _callable(*args, **kwargs)
