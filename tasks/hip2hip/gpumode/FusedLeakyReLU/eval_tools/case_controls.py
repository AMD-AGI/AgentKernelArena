# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Apply manifest-declared operator state before correctness and timing.

State construction is outside timed calls. Both roles use the same manifest;
no candidate-dependent policy or RNG draws are involved.
"""
import copy
import json
import math
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def validate_controls(rows):
    slopes, scales = set(), set()
    for row in rows:
        op = row['params']['operator']
        if set(op) != {'bias', 'negative_slope', 'scale'}:
            raise ValueError('Incomplete FusedLeakyReLU operator state')
        bias = op['bias']
        if set(bias) != {'pattern', 'offset', 'step'} or bias['pattern'] != 'alternating_channel':
            raise ValueError('Unknown channel-bias definition')
        values = [op['negative_slope'], op['scale'], bias['offset'], bias['step']]
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
            raise ValueError('Operator state must be finite numeric values')
        if not 0 < op['negative_slope'] < 1 or any(v <= 0 for v in values[1:]):
            raise ValueError('Operator controls must exercise leaky activation and nonzero bias')
        slopes.add(op['negative_slope'])
        scales.add(op['scale'])
    if len(slopes) < 3 or len(scales) < 3:
        raise ValueError('Workload must exercise multiple slopes and scales')


def control_for_inputs(inputs):
    rows = json.loads((ROOT / 'workload.json').read_text())['cases']
    validate_controls(rows)
    shapes = [list(value.shape) for value in inputs]
    matches = [row for row in rows if [v['shape'] for v in row['params']['inputs']] == shapes]
    if len(matches) != 1:
        raise ValueError('Operator state requires one declared workload shape')
    return copy.deepcopy(matches[0]['params']['operator'])


def apply_control(model, control):
    bias = model.bias
    if bias.ndim != 1 or not bias.is_floating_point():
        raise ValueError('FusedLeakyReLU bias must be a floating channel vector')
    definition = control['bias']
    # bias[c] = (-1)^c * (offset + step * (c + 1)). Each channel is nonzero
    # and distinct, so omitting bias or using another channel is observable.
    channel = torch.arange(bias.numel(), device=bias.device, dtype=torch.int64)
    magnitude = definition['offset'] + definition['step'] * (channel.to(bias.dtype) + 1)
    values = torch.where(channel.remainder(2) == 0, magnitude, -magnitude)
    with torch.no_grad():
        bias.copy_(values)
    model.negative_slope = control['negative_slope']
    model.scale = control['scale']


def configure_models(models, inputs):
    control = control_for_inputs(inputs)
    for model in models:
        if model.bias.numel() < inputs[0].shape[1]:
            raise ValueError('Bias does not cover the declared channel count')
        apply_control(model, control)
    return control
