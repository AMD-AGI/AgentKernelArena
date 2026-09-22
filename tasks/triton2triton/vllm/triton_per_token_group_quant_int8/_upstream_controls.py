"""Unscored PR105 control inputs; original performance remains unchanged.
Source revision: 0acf65b3a967ef1025dbfc5fd4b415b259e3bd43
Source path: tasks/triton2triton/vllm/triton_per_token_group_quant_int8/scripts/task_runner.py
"""
import sys
import os
import json
import argparse
import importlib.util

def reference_per_token_group_quant_int8(x, group_size, eps=1e-10):
    """CPU reference for per-token-group INT8 quantization."""
    import torch
    M, N = x.shape
    x_cpu = x.cpu().float()
    int8_max = 127
    int8_min = -128
    num_groups = N // group_size
    x_q = torch.zeros((M, N), dtype=torch.int8)
    x_s = torch.zeros(M, num_groups, dtype=torch.float32)
    for row in range(M):
        for g in range(num_groups):
            start = g * group_size
            end = start + group_size
            group = x_cpu[row, start:end]
            absmax = max(group.abs().max().item(), eps)
            scale = absmax / int8_max
            x_s[row, g] = scale
            x_q[row, start:end] = (group / scale).clamp(int8_min, int8_max).to(torch.int8)
    return (x_q, x_s)

def boundary_correctness_cases(torch, device):
    """Small deterministic cases for values Gaussian inputs do not cover."""
    group_size = 96
    all_zero = torch.zeros((1, group_size), device=device, dtype=torch.float16)
    eps_pattern = torch.tensor([-0.99951171875, -0.5, -0.25, -0.003937007874015748, -0.0, 0.0, 0.003937007874015748, 0.25, 0.5, 0.99951171875, -0.125, 0.125], device=device, dtype=torch.float16)
    eps_dominated = eps_pattern.repeat(8).reshape(1, group_size)
    extrema_pattern = torch.tensor([-65504.0, 65504.0, -32752.0, 32752.0, -1024.0, 1024.0, -6.103515625e-05, 6.103515625e-05, -5.960464477539063e-08, 5.960464477539063e-08, -0.0, 0.0], device=device, dtype=torch.float16)
    boundary_pattern = torch.tensor([-1.0, -126.5 / 127.0, -126.0 / 127.0, -64.5 / 127.0, -64.0 / 127.0, -1.5 / 127.0, -1.0 / 127.0, -0.5 / 127.0, 0.5 / 127.0, 1.0 / 127.0, 1.5 / 127.0, 64.0 / 127.0, 64.5 / 127.0, 126.0 / 127.0, 126.5 / 127.0, 1.0], device=device, dtype=torch.float16)
    extrema_and_boundaries = torch.cat((extrema_pattern.repeat(8), boundary_pattern.repeat(6))).reshape(1, 2 * group_size)
    return [('all_zero_g96', all_zero, group_size, 1.0), ('eps_dominated_g96', eps_dominated, group_size, 1.0), ('fp16_extrema_and_quant_boundaries_g96', extrema_and_boundaries, group_size, 1e-10)]
EXTRA_CASES = [{'name': 'all_zero_g96'}, {'name': 'eps_dominated_g96'}, {'name': 'fp16_extrema_and_quant_boundaries_g96'}]

def run_control(index, load_module):
    if type(index) is not int or not 0 <= index < len(EXTRA_CASES):
        raise ValueError('Unknown upstream correctness control')
    import torch
    try:
        mod = load_module()
    except Exception as e:
        return (False, f'Failed to load module: {e}')
    device = 'cuda'
    cases = []
    cases.extend(boundary_correctness_cases(torch, device))
    for _extra_index, (case_name, x, group_size, eps) in enumerate(cases):
        if _extra_index + 0 != index:
            continue
        M, N = x.shape
        try:
            x_q, x_s = mod.per_token_group_quant_int8(x, group_size, eps=eps)
            torch.cuda.synchronize()
            ref_q, ref_s = reference_per_token_group_quant_int8(x, group_size, eps=eps)
            ref_q = ref_q.to(device)
            ref_s = ref_s.to(device)
            if not torch.allclose(x_s, ref_s, atol=0.0001, rtol=0.001):
                max_diff = (x_s - ref_s).abs().max().item()
                return (False, f'Case {case_name} (M={M}, N={N}, G={group_size}): scale max diff = {max_diff:.6f}')
            if not torch.allclose(x_q.float(), ref_q.float(), atol=1.0, rtol=0.0):
                max_diff = (x_q.float() - ref_q.float()).abs().max().item()
                return (False, f'Case {case_name} (M={M}, N={N}, G={group_size}): quant max diff = {max_diff:.1f}')
        except Exception as e:
            return (False, f'Case {case_name} (M={M}, N={N}, G={group_size}): exception: {e}')
    return (True, None)
