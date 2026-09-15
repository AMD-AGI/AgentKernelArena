"""Additional unscored public-branch controls from PR105.

Source revision: 0acf65b3a967ef1025dbfc5fd4b415b259e3bd43
Source path: tasks/triton2triton/vllm/triton_scaled_mm/scripts/task_runner.py
Original scored inputs, gates and timing remain in scripts/task_runner.py.
"""
import sys
import os
import json
import argparse
import importlib.util
PERFORMANCE_SHAPES = [(32, 64, 64, True, True, False), (64, 128, 128, True, True, True), (128, 256, 256, False, False, False), (256, 512, 512, True, True, True), (64, 256, 128, True, False, False)]
CORRECTNESS_CASES = [(*shape, 'float16') for shape in PERFORMANCE_SHAPES] + [(31, 577, 8192, False, True, False, 'int8'), (33, 513, 67, False, True, True, 'float16'), (65, 515, 131, False, False, True, 'float16'), (129, 769, 257, True, False, True, 'float16')]

def reference_scaled_mm(input_t, weight, scale_a, scale_b, out_dtype, bias=None):
    """CPU reference: (input * scale_a) @ (weight * scale_b) + bias"""
    import torch
    a = input_t.float()
    b = weight.float()
    sa = scale_a.float()
    sb = scale_b.float()
    result = sa * a @ b
    result = result * sb.reshape(1, -1)
    if bias is not None:
        result = result + bias.float()
    return result.to(out_dtype)
EXTRA_CASES = [(31, 577, 8192, False, True, False, 'int8'), (33, 513, 67, False, True, True, 'float16'), (65, 515, 131, False, False, True, 'float16'), (129, 769, 257, True, False, True, 'float16')]

def run_control(index, load_module):
    if type(index) is not int or not 0 <= index < len(EXTRA_CASES):
        raise ValueError('Unknown upstream correctness control')
    import torch
    try:
        mod = load_module()
    except Exception as e:
        return (False, f'Failed to load module: {e}')
    device = 'cuda'
    out_dtype = torch.float16
    for i, case in enumerate(EXTRA_CASES, start=5):
        if i != index + 5:
            continue
        M, K, N, per_tok_a, per_ch_b, has_bias, input_dtype_name = case
        try:
            torch.manual_seed(42 + i)
            input_dtype = getattr(torch, input_dtype_name)
            if input_dtype == torch.int8:
                input_t = torch.randint(-4, 5, (M, K), device=device, dtype=input_dtype)
                weight = torch.randint(-4, 5, (K, N), device=device, dtype=input_dtype)
            else:
                input_t = torch.randn(M, K, device=device, dtype=input_dtype) * 0.1
                weight = torch.randn(K, N, device=device, dtype=input_dtype) * 0.1
            if per_tok_a:
                scale_a = torch.rand(M, 1, device=device, dtype=torch.float32) * 2 + 0.5
            else:
                scale_a = torch.rand(1, 1, device=device, dtype=torch.float32) * 2 + 0.5
            if per_ch_b:
                scale_b = torch.rand(N, 1, device=device, dtype=torch.float32) * 2 + 0.5
            else:
                scale_b = torch.rand(1, 1, device=device, dtype=torch.float32) * 2 + 0.5
            bias = torch.randn(N, device=device, dtype=out_dtype) * 0.1 if has_bias else None
            result = mod.triton_scaled_mm(input_t, weight, scale_a, scale_b, out_dtype, bias=bias)
            torch.cuda.synchronize()
            ref = reference_scaled_mm(input_t, weight, scale_a, scale_b, out_dtype, bias=bias)
            if not torch.allclose(result, ref, atol=0.01, rtol=0.01):
                max_diff = (result - ref).abs().max().item()
                return (False, f'Shape {i + 1} (M={M}, K={K}, N={N}, dtype={input_dtype_name}): max diff = {max_diff:.6f}')
        except Exception as e:
            return (False, f'Shape {i + 1} (M={M}, K={K}, N={N}, dtype={input_dtype_name}): exception: {e}')
    return (True, None)
