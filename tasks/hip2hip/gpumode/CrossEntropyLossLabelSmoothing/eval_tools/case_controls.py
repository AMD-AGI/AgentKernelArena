# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Declared final-axis probability targets and smoothing state, outside timing."""
import copy
import json
import math
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]


def validate_controls(rows):
    values = []
    for row in rows:
        op = row['params']['operator']
        if set(op) != {'class_axis', 'smooth_eps', 'smooth_dist'} or op['class_axis'] != -1:
            raise ValueError('Loss controls require the original final class axis')
        epsilon = op['smooth_eps']
        if type(epsilon) not in (float, int) or not math.isfinite(epsilon) or not 0 <= epsilon < 1:
            raise ValueError('Invalid smoothing epsilon')
        if op['smooth_dist'] != 'normalized_class_ramp':
            raise ValueError('Loss controls require the declared nonuniform smoothing distribution')
        values.append(epsilon)
    if not any(v == 0 for v in values) or len({v for v in values if v > 0}) < 3:
        raise ValueError('Loss coverage needs unsmoothed and varied nonzero smoothing states')


def control_for_inputs(inputs):
    rows = json.loads((ROOT / 'workload.json').read_text())['cases']
    validate_controls(rows)
    shapes = [list(value.shape) for value in inputs]
    matches = [row for row in rows if [v['shape'] for v in row['params']['inputs']] == shapes]
    if len(matches) != 1:
        raise ValueError('Smoothing state requires one declared workload shape')
    return copy.deepcopy(matches[0]['params']['operator'])


def apply_control(model, control, inputs):
    logits, target = inputs
    if logits.shape != target.shape or not target.is_floating_point():
        raise ValueError('Scored loss requires matching floating probability targets')
    model.smooth_eps = control['smooth_eps']
    classes = logits.shape[-1]
    ramp = torch.arange(1, classes + 1, device=logits.device, dtype=logits.dtype)
    dist = (ramp / ramp.sum()).expand_as(logits).contiguous()
    # Register state so the timed observer protects/restores the actual buffer.
    if 'smooth_dist' not in dict(model.named_buffers()):
        delattr(model, 'smooth_dist')
        model.register_buffer('smooth_dist', dist)
    else:
        model.smooth_dist = dist


def configure_models(models, inputs):
    control = control_for_inputs(inputs)
    for model in models:
        apply_control(model, control, inputs)
    return control


def reference(logits, target, epsilon, distribution):
    """Independent log-sum-exp formula for the scored mean soft-target loss."""
    if logits.shape != target.shape or distribution.shape != target.shape:
        raise ValueError('Loss reference needs full matching class distributions')
    if not torch.isfinite(logits).all() or not torch.isfinite(target).all():
        raise ValueError('Loss inputs must be finite')
    if (target < 0).any() or not torch.allclose(target.sum(-1), torch.ones_like(target[..., 0]), rtol=1e-5, atol=1e-6):
        raise ValueError('Soft targets must be probabilities over the final axis')
    if (distribution < 0).any() or not torch.allclose(distribution.sum(-1), torch.ones_like(target[..., 0]), rtol=1e-5, atol=1e-6):
        raise ValueError('Smoothing distribution must normalize over the final axis')
    # FP64 reference arithmetic avoids sharing the loss implementation/reduction.
    values = logits.double()
    shifted = values - values.amax(-1, keepdim=True)
    log_probs = shifted - shifted.exp().sum(-1, keepdim=True).log()
    mixed = (1.0 - epsilon) * target.double() + epsilon * distribution.double()
    return -(mixed * log_probs).sum(-1).mean().to(logits.dtype)


def self_test(module, functional, model_class):
    # Different non-class and class dimensions expose a wrong reduction axis.
    logits = torch.tensor([[[[0., math.log(2), math.log(4)], [math.log(4), 0., math.log(2)]]]])
    target = torch.tensor([[[[1., 0., 0.], [0., 1., 0.]]]])
    ramp = torch.tensor([1/6, 2/6, 3/6]).expand_as(logits).contiguous()
    expected_plain = torch.tensor(math.log(7), dtype=logits.dtype)
    expected_smooth = torch.tensor(math.log(7) - .3 * math.log(2) * 13/12, dtype=logits.dtype)
    for eps, expected in ((0., expected_plain), (.3, expected_smooth)):
        oracle = reference(logits, target, eps, ramp)
        torch.testing.assert_close(oracle, expected, rtol=1e-4, atol=1e-5)
        for implementation in (module, functional):
            model = getattr(implementation, model_class)(smooth_eps=eps, smooth_dist=ramp)
            original = target.clone()
            actual = model(logits, target)
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(target, original, rtol=0, atol=0)
    if torch.allclose(expected_plain, expected_smooth, rtol=1e-4, atol=1e-5):
        raise ValueError('Smoothing control must reject ignoring epsilon')
