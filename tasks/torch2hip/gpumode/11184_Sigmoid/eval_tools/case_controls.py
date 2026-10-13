# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Protect the original scored sigmoid parameters from candidate mutation."""


def assert_declared_control(model, _inputs):
    if type(model.a) not in (int, float) or type(model.max) not in (int, float) or (
        model.a != 1 or model.max != 10
    ):
        raise ValueError('Operator changed the original scored sigmoid parameters')
