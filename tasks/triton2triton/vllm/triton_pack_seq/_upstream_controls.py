"""Additional unscored public-branch controls from PR105.

Source revision: 0acf65b3a967ef1025dbfc5fd4b415b259e3bd43
Source path: tasks/triton2triton/vllm/triton_pack_seq/scripts/task_runner.py
Original scored inputs, gates and timing remain in scripts/task_runner.py.
The first three controls come from that upstream revision. The two all-empty
controls were added during final PR107 review to cover the public zero-token
domain, including higher-rank features.
"""
import sys
import os
import json
import argparse
import importlib.util
ADDITIONAL_CORRECTNESS_CASES = [{'name': 'default_pad_odd_d_float32', 'lengths': [17, 5, 9], 'feature_shape': (70,), 'dtype': 'float32'}, {'name': 'zero_lengths_long_time_bfloat16', 'lengths': [0, 70, 13, 0], 'feature_shape': (65,), 'dtype': 'bfloat16', 'pad_value': 2.5}, {'name': 'multidimensional_float16', 'lengths': [9, 0, 4], 'feature_shape': (3, 5), 'dtype': 'float16', 'pad_value': -1.25}, {'name': 'all_empty_float16', 'lengths': [0, 0, 0], 'feature_shape': (65,), 'dtype': 'float16'}, {'name': 'all_empty_multidimensional_bfloat16', 'lengths': [0, 0], 'feature_shape': (3, 5), 'dtype': 'bfloat16', 'pad_value': 2.5}]

def reference_pack_seq(x, lengths, pad_value=-float('inf')):
    """CPU/PyTorch reference for pack_seq."""
    import torch
    B = len(lengths)
    Lmax = max(lengths)
    out = torch.full((B, Lmax) + tuple(x.shape[1:]), pad_value, device=x.device, dtype=x.dtype)
    offset = 0
    for b in range(B):
        seq_len = lengths[b]
        out[b, :seq_len] = x[offset:offset + seq_len]
        offset += seq_len
    return out
EXTRA_CASES = [{'name': 'default_pad_odd_d_float32', 'lengths': [17, 5, 9], 'feature_shape': (70,), 'dtype': 'float32'}, {'name': 'zero_lengths_long_time_bfloat16', 'lengths': [0, 70, 13, 0], 'feature_shape': (65,), 'dtype': 'bfloat16', 'pad_value': 2.5}, {'name': 'multidimensional_float16', 'lengths': [9, 0, 4], 'feature_shape': (3, 5), 'dtype': 'float16', 'pad_value': -1.25}, {'name': 'all_empty_float16', 'lengths': [0, 0, 0], 'feature_shape': (65,), 'dtype': 'float16'}, {'name': 'all_empty_multidimensional_bfloat16', 'lengths': [0, 0], 'feature_shape': (3, 5), 'dtype': 'bfloat16', 'pad_value': 2.5}]

def run_control(index, load_module):
    if type(index) is not int or not 0 <= index < len(EXTRA_CASES):
        raise ValueError('Unknown upstream correctness control')
    import torch
    try:
        mod = load_module()
    except Exception as e:
        return (False, f'Failed to load module: {e}')
    device = 'cuda'
    dtype = torch.float16
    for i, case in enumerate(EXTRA_CASES, start=0):
        if i != index + 0:
            continue
        try:
            torch.manual_seed(100 + i)
            lengths_list = case['lengths']
            N = sum(lengths_list)
            x = torch.randn((N,) + case['feature_shape'], device=device, dtype=getattr(torch, case['dtype']))
            lengths = torch.tensor(lengths_list, device=device, dtype=torch.int32)
            if 'pad_value' in case:
                pad_value = case['pad_value']
                result = mod.pack_seq(x, lengths, pad_value=pad_value)
                ref = reference_pack_seq(x, lengths_list, pad_value=pad_value)
            else:
                result = mod.pack_seq(x, lengths)
                ref = reference_pack_seq(x, lengths_list)
            torch.cuda.synchronize()
            if not torch.allclose(result, ref, atol=0.001, rtol=0.001):
                return (False, f'Case {case['name']}: output differs from reference')
        except Exception as e:
            return (False, f'Case {case['name']}: exception: {e}')
    return (True, None)
