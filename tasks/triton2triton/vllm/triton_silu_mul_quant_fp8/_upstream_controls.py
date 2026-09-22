"""Additional unscored public-branch controls from PR105.

Source revision: 0acf65b3a967ef1025dbfc5fd4b415b259e3bd43
Source path: tasks/triton2triton/vllm/triton_silu_mul_quant_fp8/scripts/task_runner.py
Original scored inputs, gates and timing remain in scripts/task_runner.py.
"""
import sys
import os
import json
import math
import argparse
import importlib.util
CORRECTNESS_CASES = [{'name': 'bfloat16_ue8m0', 'shape': (128, 256), 'input_dtype': 'bfloat16', 'input_kind': 'random', 'inject_wide_value': True, 'eps': 1e-06, 'use_ue8m0': True}, {'name': 'float32_edges_preallocated_output', 'shape': (128, 512), 'input_dtype': 'float32', 'input_kind': 'edge_values', 'eps': 0.0001, 'preallocate_output': True, 'exact_quantized': True}]

def reference_silu_mul_quant_fp8(input_t, fp8_dtype, *, eps=1e-10, use_ue8m0=False):
    """CPU reference: silu(x[:,:N/2]) * x[:,N/2:], then quantize per group."""
    import torch
    GROUP_SIZE = 128
    M, N = input_t.shape
    N_2 = N // 2
    x = input_t.cpu().float()
    gate = x[:, :N_2]
    up = x[:, N_2:]
    silu_out = gate / (1.0 + torch.exp(-gate))
    y = silu_out * up
    if fp8_dtype == torch.float8_e4m3fnuz:
        fp8_min, fp8_max = (-240.0, 240.0)
    else:
        finfo = torch.finfo(fp8_dtype)
        fp8_min, fp8_max = (finfo.min, finfo.max)
    num_groups = N_2 // GROUP_SIZE
    y_q = torch.zeros_like(y)
    y_s = torch.zeros(M, num_groups, dtype=torch.float32)
    for row in range(M):
        for g in range(num_groups):
            start = g * GROUP_SIZE
            end = start + GROUP_SIZE
            group = y[row, start:end]
            absmax = max(group.abs().max().item(), eps)
            scale_raw = absmax / fp8_max
            scale = 2.0 ** math.ceil(math.log2(scale_raw)) if use_ue8m0 else scale_raw
            y_s[row, g] = scale
            y_q[row, start:end] = (group / scale).clamp(fp8_min, fp8_max)
    return (y_q.to(fp8_dtype), y_s)

def _make_correctness_input(torch, case):
    """Create deterministic inputs for the targeted correctness cases."""
    dtype = getattr(torch, case['input_dtype'])
    shape = case['shape']
    if case['input_kind'] == 'random':
        torch.manual_seed(123)
        x = torch.randn(shape, device='cuda', dtype=dtype)
        if case.get('inject_wide_value', False):
            N_2 = shape[1] // 2
            x[:, 0] = 100000.0
            x[:, N_2] = 0.0001
        return x
    x = torch.zeros(shape, device='cuda', dtype=dtype)
    N_2 = shape[1] // 2
    x[:, 0] = 1e-20
    x[:, N_2] = 1.0
    x[:, 1] = -1e-20
    x[:, N_2 + 1] = 1.0
    x[:, 2] = 100000.0
    x[:, N_2 + 2] = 0.064
    x[:, 3] = 100000.0
    x[:, N_2 + 3] = -0.064
    return x

def _check_outputs(torch, case_name, y_q, y_s, ref_q, ref_s, *, exact_quantized=False, scale_atol=0.01, scale_rtol=0.1):
    """Compare scales and quantized results without changing task tolerances."""
    if y_q.shape != ref_q.shape or y_s.shape != ref_s.shape:
        return f'{case_name}: output shapes {(tuple(y_q.shape), tuple(y_s.shape))} do not match expected {(tuple(ref_q.shape), tuple(ref_s.shape))}'
    if y_q.dtype != ref_q.dtype or y_s.dtype != ref_s.dtype:
        return f'{case_name}: output dtypes {(y_q.dtype, y_s.dtype)} do not match expected {(ref_q.dtype, ref_s.dtype)}'
    if not torch.allclose(y_s, ref_s, atol=scale_atol, rtol=scale_rtol):
        max_diff = (y_s - ref_s).abs().max().item()
        return f'{case_name}: scale max diff = {max_diff:.6g}'
    if exact_quantized and (not torch.equal(y_q.float(), ref_q.float())):
        max_diff = (y_q.float() - ref_q.float()).abs().max().item()
        return f'{case_name}: quantized max diff = {max_diff:.6g}'
    GROUP_SIZE = 128
    y_dq = y_q.float() * y_s.repeat_interleave(GROUP_SIZE, dim=-1)
    ref_dq = ref_q.float() * ref_s.repeat_interleave(GROUP_SIZE, dim=-1)
    if not torch.allclose(y_dq, ref_dq, atol=0.5, rtol=0.1):
        max_diff = (y_dq - ref_dq).abs().max().item()
        return f'{case_name}: dequant max diff = {max_diff:.6f}'
    return None
EXTRA_CASES = [{'name': 'bfloat16_ue8m0', 'shape': (128, 256), 'input_dtype': 'bfloat16', 'input_kind': 'random', 'inject_wide_value': True, 'eps': 1e-06, 'use_ue8m0': True}, {'name': 'float32_edges_preallocated_output', 'shape': (128, 512), 'input_dtype': 'float32', 'input_kind': 'edge_values', 'eps': 0.0001, 'preallocate_output': True, 'exact_quantized': True}]

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
    for _control_index, case in enumerate(EXTRA_CASES, start=0):
        if _control_index != index + 0:
            continue
        case_name = case['name']
        try:
            x = _make_correctness_input(torch, case)
            M, N = x.shape
            eps = case.get('eps', 1e-10)
            use_ue8m0 = case.get('use_ue8m0', False)
            output = None
            if case.get('preallocate_output', False):
                output = torch.full((M, N // 2), 240.0, device=device, dtype=fp8_dtype)
            y_q, y_s = mod.silu_mul_per_token_group_quant_fp8_colmajor(x, output=output, use_ue8m0=use_ue8m0, eps=eps)
            torch.cuda.synchronize()
            if output is not None and y_q.data_ptr() != output.data_ptr():
                return (False, f'{case_name}: returned a different output buffer')
            ref_q, ref_s = reference_silu_mul_quant_fp8(x, fp8_dtype, eps=eps, use_ue8m0=use_ue8m0)
            ref_q = ref_q.to(device)
            ref_s = ref_s.to(device)
            error = _check_outputs(torch, case_name, y_q, y_s, ref_q, ref_s, exact_quantized=case.get('exact_quantized', False), scale_atol=1e-08, scale_rtol=0.0001)
            if error:
                return (False, error)
        except Exception as e:
            return (False, f'{case_name}: exception: {e}')
    return (True, None)
