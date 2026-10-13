# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Nonidentity affine state for every declared normalization case."""
import copy
import json
import math
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]


def validate_controls(rows):
    amplitudes = set()
    for row in rows:
        op = row['params']['operator']
        if set(op) != {'gamma', 'beta', 'beta_amplitude'}:
            raise ValueError('Incomplete normalization affine control')
        if op['gamma'] != '0.5_plus_flat_index_fraction' or op['beta'] != 'alternating_flat_ramp':
            raise ValueError('Unsupported normalization affine pattern')
        value = op['beta_amplitude']
        if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
            raise ValueError('Normalization needs nonzero finite beta')
        amplitudes.add(value)
    if len(amplitudes) < 3:
        raise ValueError('Normalization needs varied affine states')


def affine(model):
    return (model.gamma, model.beta) if hasattr(model, 'gamma') else (model.ln.weight, model.ln.bias)


def apply_control(model, control):
    gamma, beta = affine(model)
    if gamma.shape != beta.shape or gamma.numel() < 2:
        raise ValueError('Normalization affine tensors must cover all normalized elements')
    index = torch.arange(1, gamma.numel() + 1, dtype=gamma.dtype, device=gamma.device)
    sign = torch.where(index.remainder(2) == 0, 1., -1.)
    with torch.no_grad():
        gamma.copy_((.5 + index / (gamma.numel() + 1)).reshape_as(gamma))
        beta.copy_((sign * control['beta_amplitude'] * (1 + index / gamma.numel())).reshape_as(beta))


def configure_models(models, inputs):
    rows = json.loads((ROOT / 'workload.json').read_text())['cases']
    validate_controls(rows)
    shapes = [list(value.shape) for value in inputs]
    matches = [row for row in rows if [v['shape'] for v in row['params']['inputs']] == shapes]
    if len(matches) != 1:
        raise ValueError('Affine state needs exactly one declared shape')
    control = copy.deepcopy(matches[0]['params']['operator'])
    for model in models:
        gamma, _ = affine(model)
        if list(inputs[0].shape[-gamma.ndim:]) != list(gamma.shape):
            raise ValueError('Affine state must match normalized input dimensions')
        apply_control(model, control)
    return control


def self_test(module, functional, model_class):
    ordinary = model_class == 'LayerNorm'
    # Independent four-number mean/variance answer, with both nonidentity scale
    # and nonzero shift. Keep the task's actual sample-vs-population convention.
    values = [1., 2., 4., 8.]
    mean = sum(values) / 4
    variance = sum((v - mean)**2 for v in values) / (4 if ordinary else 3)
    shape = (1, 2, 2) if ordinary else (1, 4)
    x = torch.tensor(values).reshape(shape)
    gamma = torch.tensor([.7, .9, 1.1, 1.3]).reshape(shape[1:])
    beta = torch.tensor([-.15625, .1875, -.21875, .25]).reshape(shape[1:])
    for implementation in (module, functional):
        model = getattr(implementation, model_class)((2, 2) if ordinary else 4).eval()
        apply_control(model, {'beta_amplitude': .125})
        eps = model.ln.eps if ordinary else model.epsilon
        denominator = math.sqrt(variance + eps) if ordinary else math.sqrt(variance) + eps
        normalized = torch.tensor([(v - mean) / denominator for v in values]).reshape(shape)
        expected = normalized * gamma + beta
        actual = model(x)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        for wrong in (normalized, normalized + beta, normalized * gamma, torch.zeros_like(expected)):
            if torch.allclose(wrong, expected, rtol=1e-4, atol=1e-5):
                raise ValueError('Affine control did not distinguish an omitted parameter')
