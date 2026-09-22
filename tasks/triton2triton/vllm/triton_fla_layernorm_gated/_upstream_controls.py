"""Unscored PR105 public-branch controls, ported from pinned main.

The original runner still owns every existing correctness and performance path.
These additional calls use the current checked candidate loader. No benchmark
helper or original workload is replaced.
Source revision: 0acf65b3a967ef1025dbfc5fd4b415b259e3bd43
Source path: tasks/triton2triton/vllm/triton_fla_layernorm_gated/scripts/task_runner.py
"""
import sys
import os
import json
import argparse
import importlib.util
TEST_CASES = [(256, 128, 'swish', True, True, False), (192, 96, 'sigmoid', True, True, True), (128, 64, 'swish', False, True, True), (160, 80, 'swish', True, False, False), (64, 48, 'sigmoid', False, False, True)]
CORRECTNESS_TEST_CASES = [(*case, 'float32', 1e-05) for case in TEST_CASES] + [(1, 1, 'silu', False, True, True, 'float16', 1e-06), (7, 31, 'sigmoid', True, False, False, 'bfloat16', 0.0001), (9, 129, 'swish', False, False, True, 'float16', 0.001), (33, 1025, 'sigmoid', True, True, False, 'bfloat16', 0.01)]

def reference(x, g, weight=None, bias=None, activation='swish', eps=1e-05, is_rms_norm=True):
    import torch
    x_f = x.float().cpu()
    g_f = g.float().cpu()
    mean = None
    if is_rms_norm:
        var = (x_f * x_f).mean(dim=-1, keepdim=True)
        rstd = 1.0 / torch.sqrt(var + eps)
        x_hat = x_f * rstd
    else:
        mean = x_f.mean(dim=-1, keepdim=True)
        var = ((x_f - mean) ** 2).mean(dim=-1, keepdim=True)
        rstd = 1.0 / torch.sqrt(var + eps)
        x_hat = (x_f - mean) * rstd
    if weight is not None:
        x_hat = x_hat * weight.float().cpu()
    if bias is not None:
        x_hat = x_hat + bias.float().cpu()
    if activation in ('swish', 'silu'):
        y = x_hat * g_f * torch.sigmoid(g_f)
    elif activation == 'sigmoid':
        y = x_hat * torch.sigmoid(g_f)
    else:
        y = x_hat
    mean = mean.squeeze(-1) if mean is not None else None
    return (y.to(x.dtype), mean, rstd.squeeze(-1))

def gen_inputs(seed, test_case, device):
    import torch
    torch.manual_seed(seed)
    T, D, activation, is_rms_norm, has_weight, has_bias, dtype_name, eps = test_case
    dtype = getattr(torch, dtype_name)
    x = torch.randn(T, D, device=device, dtype=dtype)
    g = torch.randn(T, D, device=device, dtype=dtype)
    w = torch.randn(D, device=device, dtype=dtype) if has_weight else None
    b = torch.randn(D, device=device, dtype=dtype) if has_bias else None
    kwargs = {'weight': w, 'bias': b, 'activation': activation, 'eps': eps, 'is_rms_norm': is_rms_norm}
    return ((x, g), kwargs)
EXTRA_CASES = [(1, 1, 'silu', False, True, True, 'float16', 1e-06), (7, 31, 'sigmoid', True, False, False, 'bfloat16', 0.0001), (9, 129, 'swish', False, False, True, 'float16', 0.001), (33, 1025, 'sigmoid', True, True, False, 'bfloat16', 0.01)]

def run_control(index, load_module):
    if type(index) is not int or not 0 <= index < len(EXTRA_CASES):
        raise ValueError('Unknown upstream correctness control')
    import torch
    try:
        mod = load_module()
    except Exception as e:
        return (False, f'Failed to load module: {e}')
    device = 'cuda'
    for i, test_case in enumerate(CORRECTNESS_TEST_CASES):
        if i != index + 5:
            continue
        try:
            args, kwargs = gen_inputs(42 + i, test_case, device)
            args_cpu = tuple((a.cpu() if isinstance(a, torch.Tensor) else a for a in args))
            result_tuple = mod.layer_norm_gated_fwd(*args, **kwargs)
            if not isinstance(result_tuple, tuple) or len(result_tuple) != 3:
                return (False, f'Case {i + 1} {test_case}: expected (y, mean, rstd) tuple')
            result, mean, rstd = result_tuple
            ref_result, ref_mean, ref_rstd = reference(args_cpu[0], args_cpu[1], weight=kwargs['weight'], bias=kwargs['bias'], activation=kwargs['activation'], eps=kwargs['eps'], is_rms_norm=kwargs['is_rms_norm'])
            for name, actual, expected, expected_dtype in (('y', result, ref_result, args[0].dtype), ('mean', mean, ref_mean, torch.float32), ('rstd', rstd, ref_rstd, torch.float32)):
                if expected is None:
                    if actual is not None:
                        return (False, f'Case {i + 1} {test_case}: expected {name}=None')
                    continue
                if actual is None:
                    return (False, f'Case {i + 1} {test_case}: {name} is None')
                if actual.dtype != expected_dtype:
                    return (False, f'Case {i + 1} {test_case}: {name} dtype {actual.dtype} != {expected_dtype}')
                actual_f = actual.float().cpu()
                expected_f = expected.float()
                if actual_f.shape != expected_f.shape:
                    return (False, f'Case {i + 1} {test_case}: {name} shape {tuple(actual_f.shape)} != {tuple(expected_f.shape)}')
                if not torch.allclose(actual_f, expected_f, atol=0.001, rtol=0.001):
                    max_diff = (actual_f - expected_f).abs().max().item()
                    return (False, f'Case {i + 1} {test_case}: {name} max diff = {max_diff:.6f}')
            torch.cuda.synchronize()
        except Exception as e:
            return (False, f'Case {i + 1} {test_case}: exception: {e}')
    return (True, None)
