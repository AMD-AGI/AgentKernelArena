# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Manifest-controlled channel scale and scalar bias, outside measured calls."""
import copy
import json
import math
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]


def validate_controls(rows):
    biases = set()
    for row in rows:
        op = row['params']['operator']
        if set(op) != {'scale', 'bias'} or set(op['scale']) != {'pattern', 'offset', 'span'}:
            raise ValueError('Incomplete InnerProd affine state')
        scale = op['scale']
        values = (scale['offset'], scale['span'], op['bias'])
        if scale['pattern'] != 'channel_ramp' or any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
            raise ValueError('Invalid InnerProd affine state')
        if scale['offset'] <= 0 or scale['span'] <= 0 or op['bias'] == 0:
            raise ValueError('Affine controls require varying scale and nonzero bias')
        biases.add(op['bias'])
    if len(biases) < 3:
        raise ValueError('Affine controls require multiple nonzero bias values')


def control_for_inputs(inputs):
    rows = json.loads((ROOT / 'workload.json').read_text())['cases']
    validate_controls(rows)
    shapes = [list(value.shape) for value in inputs]
    matches = [row for row in rows if [v['shape'] for v in row['params']['inputs']] == shapes]
    if len(matches) != 1:
        raise ValueError('Affine state requires one declared workload shape')
    return copy.deepcopy(matches[0]['params']['operator'])


def apply_control(model, control):
    if model.scale.ndim != 1 or model.bias.numel() != 1:
        raise ValueError('InnerProd needs a channel scale vector and scalar bias')
    if not model.scale.is_floating_point() or not model.bias.is_floating_point():
        raise ValueError('InnerProd affine state must be floating point')
    scale = model.scale
    ramp = torch.arange(1, scale.numel() + 1, device=scale.device, dtype=scale.dtype)
    definition = control['scale']
    with torch.no_grad():
        scale.copy_(definition['offset'] + definition['span'] * ramp / scale.numel())
        model.bias.fill_(control['bias'])


def configure_models(models, inputs):
    control = control_for_inputs(inputs)
    for model in models:
        if model.scale.numel() != inputs[1].shape[1]:
            raise ValueError('Scale must cover every input channel')
        apply_control(model, control)
    return control
