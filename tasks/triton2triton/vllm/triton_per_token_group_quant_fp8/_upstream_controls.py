"""Additional unscored public-branch controls from PR105.

Source revision: 0acf65b3a967ef1025dbfc5fd4b415b259e3bd43
Source path: tasks/triton2triton/vllm/triton_per_token_group_quant_fp8/scripts/task_runner.py
Original scored inputs, gates and timing remain in scripts/task_runner.py.
"""
import sys
import os
import json
import math
import argparse
import importlib.util
CORRECTNESS_CASES = [{'name': 'zeros', 'shape': (1, 128), 'group_size': 32, 'input_kind': 'zeros', 'input_dtype': 'float16'}, {'name': 'tiny_custom_eps', 'shape': (2, 96), 'group_size': 48, 'input_kind': 'patterned', 'amplitudes': (5e-05, 1e-08), 'input_dtype': 'float32', 'eps': 0.0001}, {'name': 'saturation_boundary_explicit_dtype', 'shape': (2, 128), 'group_size': 64, 'input_kind': 'saturation_boundary', 'input_dtype': 'float16', 'output_dtype': 'wider_fp8'}, {'name': 'mixed_dynamic_range_3d', 'shape': (2, 3, 192), 'group_size': 48, 'input_kind': 'patterned', 'amplitudes': (2 ** (-12), 2 ** (-4), 1.0, 64.0), 'input_dtype': 'bfloat16'}, {'name': 'ue8m0_non_power_of_two_group', 'shape': (3, 120), 'group_size': 40, 'input_kind': 'patterned', 'amplitudes': (0.75, 17.0, 93.0), 'input_dtype': 'float32', 'use_ue8m0': True}]

def reference_per_token_group_quant_fp8(x, group_size, fp8_dtype, fp8_min, fp8_max, eps=1e-10, use_ue8m0=False):
    """CPU reference for per-token-group FP8 quantization."""
    import torch
    N = x.shape[-1]
    leading_shape = x.shape[:-1]
    x_cpu = x.cpu().float().reshape(-1, N)
    M = x_cpu.shape[0]
    num_groups = N // group_size
    x_q = torch.zeros_like(x_cpu)
    x_s = torch.zeros(M, num_groups, dtype=torch.float32)
    for row in range(M):
        for g in range(num_groups):
            start = g * group_size
            end = start + group_size
            group = x_cpu[row, start:end]
            absmax = max(group.abs().max().item(), eps)
            scale_raw = absmax / fp8_max
            scale = 2.0 ** math.ceil(math.log2(scale_raw)) if use_ue8m0 else scale_raw
            x_s[row, g] = scale
            x_q[row, start:end] = (group / scale).clamp(fp8_min, fp8_max)
    scale_shape = leading_shape + (num_groups,)
    return (x_q.reshape(x.shape).to(fp8_dtype), x_s.reshape(scale_shape))

def _get_reference_fp8_min_max(torch, fp8_dtype):
    """Derive platform clamp limits independently of the implementation."""
    if hasattr(torch, 'float8_e4m3fnuz') and fp8_dtype == torch.float8_e4m3fnuz:
        return (-240.0, 240.0)
    finfo = torch.finfo(fp8_dtype)
    return (finfo.min, finfo.max)

def _get_wider_fp8_dtype(torch, default_dtype, fp8_max):
    """Return a supported explicit FP8 dtype that can represent the clamp range."""
    for name in ('float8_e4m3fn', 'float8_e4m3fnuz'):
        if not hasattr(torch, name):
            continue
        dtype = getattr(torch, name)
        if dtype == default_dtype or torch.finfo(dtype).max < fp8_max:
            continue
        try:
            torch.empty(1, device='cuda', dtype=dtype)
            return dtype
        except Exception:
            continue
    return default_dtype

def _make_patterned_input(torch, shape, group_size, amplitudes, dtype):
    """Build deterministic groups whose maxima span the requested amplitudes."""
    rows = math.prod(shape[:-1])
    num_groups = shape[-1] // group_size
    base = torch.linspace(-1.0, 1.0, group_size, dtype=torch.float32)
    x = torch.empty((rows, shape[-1]), dtype=torch.float32)
    for row in range(rows):
        for group in range(num_groups):
            amplitude = amplitudes[(row * num_groups + group) % len(amplitudes)]
            start = group * group_size
            x[row, start:start + group_size] = base * amplitude
    return x.reshape(shape).to(device='cuda', dtype=dtype)

def _make_correctness_input(torch, case, fp8_min, fp8_max):
    dtype = getattr(torch, case['input_dtype'])
    if case['input_kind'] == 'zeros':
        return torch.zeros(case['shape'], device='cuda', dtype=dtype)
    if case['input_kind'] == 'saturation_boundary':
        return _make_patterned_input(torch, case['shape'], case['group_size'], (fp8_max * 0.25, abs(fp8_min) * 0.25), dtype)
    return _make_patterned_input(torch, case['shape'], case['group_size'], case['amplitudes'], dtype)

def _check_outputs(torch, case_name, x_q, x_s, ref_q, ref_s, group_size, *, exact_quantized=False, scale_atol=1e-05, scale_rtol=0.001):
    if not torch.allclose(x_s, ref_s, atol=scale_atol, rtol=scale_rtol):
        max_diff = (x_s - ref_s).abs().max().item()
        return f'{case_name}: scale max diff = {max_diff:.6g}'
    if exact_quantized and (not torch.equal(x_q.float(), ref_q.float())):
        max_diff = (x_q.float() - ref_q.float()).abs().max().item()
        return f'{case_name}: quantized max diff = {max_diff:.6g}'
    x_dq = x_q.float() * x_s.repeat_interleave(group_size, dim=-1)
    ref_dq = ref_q.float() * ref_s.repeat_interleave(group_size, dim=-1)
    if not torch.allclose(x_dq, ref_dq, atol=0.1, rtol=0.1):
        max_diff = (x_dq - ref_dq).abs().max().item()
        return f'{case_name}: dequant max diff = {max_diff:.6f}'
    return None
EXTRA_CASES = [{'name': 'zeros', 'shape': (1, 128), 'group_size': 32, 'input_kind': 'zeros', 'input_dtype': 'float16'}, {'name': 'tiny_custom_eps', 'shape': (2, 96), 'group_size': 48, 'input_kind': 'patterned', 'amplitudes': (5e-05, 1e-08), 'input_dtype': 'float32', 'eps': 0.0001}, {'name': 'saturation_boundary_explicit_dtype', 'shape': (2, 128), 'group_size': 64, 'input_kind': 'saturation_boundary', 'input_dtype': 'float16', 'output_dtype': 'wider_fp8'}, {'name': 'mixed_dynamic_range_3d', 'shape': (2, 3, 192), 'group_size': 48, 'input_kind': 'patterned', 'amplitudes': (0.000244140625, 0.0625, 1.0, 64.0), 'input_dtype': 'bfloat16'}, {'name': 'ue8m0_non_power_of_two_group', 'shape': (3, 120), 'group_size': 40, 'input_kind': 'patterned', 'amplitudes': (0.75, 17.0, 93.0), 'input_dtype': 'float32', 'use_ue8m0': True}]

def run_control(index, load_module):
    if type(index) is not int or not 0 <= index < len(EXTRA_CASES):
        raise ValueError('Unknown upstream correctness control')
    import torch
    try:
        mod = load_module()
    except Exception as e:
        return (False, f'Failed to load module: {e}')
    device = 'cuda'
    fp8_dtype = mod._get_fp8_dtype()
    fp8_min, fp8_max = _get_reference_fp8_min_max(torch, fp8_dtype)
    wider_fp8_dtype = _get_wider_fp8_dtype(torch, fp8_dtype, fp8_max)
    for _control_index, case in enumerate(EXTRA_CASES, start=0):
        if _control_index != index + 0:
            continue
        case_name = case['name']
        try:
            x = _make_correctness_input(torch, case, fp8_min, fp8_max)
            output_dtype = wider_fp8_dtype if case.get('output_dtype') == 'wider_fp8' else fp8_dtype
            eps = case.get('eps', 1e-10)
            use_ue8m0 = case.get('use_ue8m0', False)
            group_size = case['group_size']
            x_q, x_s = mod.per_token_group_quant_fp8(x, group_size, eps=eps, dtype=output_dtype, use_ue8m0=use_ue8m0)
            torch.cuda.synchronize()
            if x_q.dtype != output_dtype:
                return (False, f'{case_name}: output dtype {x_q.dtype} does not match requested dtype {output_dtype}')
            ref_q, ref_s = reference_per_token_group_quant_fp8(x, group_size, output_dtype, fp8_min, fp8_max, eps=eps, use_ue8m0=use_ue8m0)
            ref_q = ref_q.to(device)
            ref_s = ref_s.to(device)
            error = _check_outputs(torch, case_name, x_q, x_s, ref_q, ref_s, group_size, exact_quantized=True, scale_atol=0.0, scale_rtol=0.0001)
            if error:
                return (False, error)
        except Exception as e:
            return (False, f'{case_name}: exception: {e}')
    return (True, None)
