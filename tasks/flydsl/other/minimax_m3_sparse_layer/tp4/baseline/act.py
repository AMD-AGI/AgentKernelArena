# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""Shared device-side gate/up activation helpers for FlyDSL MoE kernels (the swiglu subset).

Vendored from ROCm/aiter aiter/ops/flydsl/kernels/act.py at 6482901d9f2327ec7a24ce5de62bc111dd40279d
("[FlyDSL] Enable SwiGLU and tune MiniMax-M3 A16W4 MoE on gfx950 (#5395)"): ``swiglu_mul_batch``,
``_sigmoid_f32`` and ``LOG2E`` (also aiter's ``kernels_common.LOG2E``), with ``tensor_shim._to_raw`` replaced
by FlyDSL's ``as_ir_value``. The other activations are not used by this kernel and are not vendored.
"""

import flydsl.expr as fx
from flydsl.expr import rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import as_ir_value as _raw

LOG2E = 1.4426950408889634


def swiglu_mul_batch(gs, us, neg_clamp_limit):
    out = []
    for i in range(len(gs)):
        gate = -((-gs[i]).maximumf(neg_clamp_limit))
        up = (-((-us[i]).maximumf(neg_clamp_limit))).maximumf(neg_clamp_limit)
        out.append(
            gate * _sigmoid_f32(fx.Float32(1.702) * gate) * (up + fx.Float32(1.0))
        )
    return out


def _sigmoid_f32(g):
    e = fx.Float32(rocdl.exp2(T.f32, _raw(g * fx.Float32(-LOG2E))))
    return fx.Float32(rocdl.rcp(T.f32, _raw(fx.Float32(1.0) + e)))
