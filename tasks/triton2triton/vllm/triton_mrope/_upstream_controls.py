"""Additional unscored public-branch controls from PR105.

Source revision: 0acf65b3a967ef1025dbfc5fd4b415b259e3bd43
Source path: tasks/triton2triton/vllm/triton_mrope/scripts/task_runner.py
Original scored inputs, gates and timing remain in scripts/task_runner.py.
"""
import sys
import os
import json
import argparse
import importlib.util
TEST_SHAPES = [(32, 8, 8, 64, 64, [16, 8, 8]), (64, 16, 4, 64, 64, [16, 8, 8]), (128, 32, 8, 128, 64, [16, 8, 8]), (256, 16, 16, 64, 64, [16, 8, 8]), (16, 8, 2, 128, 64, [16, 8, 8])]
CORRECTNESS_CASES = [{'shape': shape, 'mrope_interleaved': False, 'dtype': 'float16'} for shape in TEST_SHAPES] + [{'shape': (17, 28, 3, 128, 128, [24, 20, 20]), 'mrope_interleaved': True, 'dtype': 'float16'}, {'shape': (1, 7, 5, 96, 96, [1, 23, 24]), 'mrope_interleaved': False, 'dtype': 'bfloat16'}, {'shape': (9, 3, 5, 64, 32, [14, 1, 1]), 'mrope_interleaved': False, 'dtype': 'float16'}]

def reference_mrope(q, k, cos, sin, mrope_section, head_size, rotary_dim, mrope_interleaved):
    """CPU/PyTorch reference for MRoPE.

    q: [num_tokens, num_q_heads * head_size]
    k: [num_tokens, num_kv_heads * head_size]
    cos: [3, num_tokens, rotary_dim // 2]
    sin: [3, num_tokens, rotary_dim // 2]
    mrope_section: [t, h, w]
    """
    import torch
    num_tokens = q.shape[0]
    n_q_head = q.shape[1] // head_size
    n_kv_head = k.shape[1] // head_size
    half_rd = rotary_dim // 2
    t_sec, h_sec, w_sec = mrope_section
    offsets = torch.arange(half_rd, device=q.device)
    if mrope_interleaved:
        h_mask = (offsets % 3 == 1) & (offsets <= 3 * h_sec)
        w_mask = (offsets % 3 == 2) & (offsets <= 3 * w_sec)
        t_mask = ~(h_mask | w_mask)
    else:
        t_end = t_sec
        h_end = t_end + h_sec
        t_mask = offsets < t_end
        h_mask = (t_end <= offsets) & (offsets < h_end)
        w_mask = h_end <= offsets
    combined_cos = torch.where(t_mask, cos[0], torch.where(h_mask, cos[1], cos[2]))
    combined_sin = torch.where(t_mask, sin[0], torch.where(h_mask, sin[1], sin[2]))
    q_out = q.clone()
    for h in range(n_q_head):
        offset = h * head_size
        x1 = q_out[:, offset:offset + half_rd].float()
        x2 = q_out[:, offset + half_rd:offset + rotary_dim].float()
        c = combined_cos.float()
        s = combined_sin.float()
        q_out[:, offset:offset + half_rd] = (x1 * c - x2 * s).to(q.dtype)
        q_out[:, offset + half_rd:offset + rotary_dim] = (x2 * c + x1 * s).to(q.dtype)
    k_out = k.clone()
    for h in range(n_kv_head):
        offset = h * head_size
        x1 = k_out[:, offset:offset + half_rd].float()
        x2 = k_out[:, offset + half_rd:offset + rotary_dim].float()
        c = combined_cos.float()
        s = combined_sin.float()
        k_out[:, offset:offset + half_rd] = (x1 * c - x2 * s).to(k.dtype)
        k_out[:, offset + half_rd:offset + rotary_dim] = (x2 * c + x1 * s).to(k.dtype)
    return (q_out, k_out)
EXTRA_CASES = [{'shape': (17, 28, 3, 128, 128, [24, 20, 20]), 'mrope_interleaved': True, 'dtype': 'float16'}, {'shape': (1, 7, 5, 96, 96, [1, 23, 24]), 'mrope_interleaved': False, 'dtype': 'bfloat16'}, {'shape': (9, 3, 5, 64, 32, [14, 1, 1]), 'mrope_interleaved': False, 'dtype': 'float16'}]

def run_control(index, load_module):
    if type(index) is not int or not 0 <= index < len(EXTRA_CASES):
        raise ValueError('Unknown upstream correctness control')
    import torch
    try:
        mod = load_module()
    except Exception as e:
        return (False, f'Failed to load module: {e}')
    device = 'cuda'
    for i, case in enumerate(EXTRA_CASES, start=5):
        if i != index + 5:
            continue
        num_tokens, n_qh, n_kh, head_size, rotary_dim, mrope_section = case['shape']
        mrope_interleaved = case['mrope_interleaved']
        dtype = getattr(torch, case['dtype'])
        try:
            torch.manual_seed(42 + i)
            q = torch.randn(num_tokens, n_qh * head_size, device=device, dtype=dtype)
            k = torch.randn(num_tokens, n_kh * head_size, device=device, dtype=dtype)
            cos = torch.randn(3, num_tokens, rotary_dim // 2, device=device, dtype=dtype)
            sin = torch.randn(3, num_tokens, rotary_dim // 2, device=device, dtype=dtype)
            q_ref = q.clone()
            k_ref = k.clone()
            q_triton = q.clone()
            k_triton = k.clone()
            mod.triton_mrope(q_triton, k_triton, cos, sin, mrope_section, head_size, rotary_dim, mrope_interleaved)
            torch.cuda.synchronize()
            q_expected, k_expected = reference_mrope(q_ref, k_ref, cos, sin, mrope_section, head_size, rotary_dim, mrope_interleaved)
            if not torch.allclose(q_triton, q_expected, atol=0.01, rtol=0.01):
                max_diff = (q_triton - q_expected).abs().max().item()
                return (False, f'Shape {i + 1} q mismatch: max diff = {max_diff:.6f}')
            if not torch.allclose(k_triton, k_expected, atol=0.01, rtol=0.01):
                max_diff = (k_triton - k_expected).abs().max().item()
                return (False, f'Shape {i + 1} k mismatch: max diff = {max_diff:.6f}')
        except Exception as e:
            return (False, f'Shape {i + 1}: exception: {e}')
    return (True, None)
