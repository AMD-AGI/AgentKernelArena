"""Unscored PR105 public-branch controls, ported from pinned main.

The original runner still owns every existing correctness and performance path.
These additional calls use the current checked candidate loader. No benchmark
helper or original workload is replaced.
Source revision: 0acf65b3a967ef1025dbfc5fd4b415b259e3bd43
Source path: tasks/triton2triton/vllm/triton_unpack_seq/scripts/task_runner.py
"""
import sys
import os
import json
import argparse
import importlib.util
TEST_SHAPES = [(4, [8, 12, 6, 10], 64), (2, [32, 16], 128), (8, [4, 8, 2, 16, 6, 10, 3, 7], 64), (3, [64, 32, 48], 256), (6, [10, 20, 15, 5, 25, 12], 128)]
CORRECTNESS_CASES = [{'packed_shape': (B, max(lengths_list), D), 'lengths': lengths_list} for B, lengths_list, D in TEST_SHAPES] + [{'packed_shape': (5, 17, 70), 'lengths': [0, 7, 8, 9, 17], 'dtype': 'float32', 'lengths_dtype': 'int64', 'block_t': 8, 'block_d': 32}, {'packed_shape': (3, 9, 2, 3, 5), 'lengths': [9, 0, 4], 'block_t': 16, 'block_d': 16}]

def reference_unpack_seq(packed, lengths_list):
    """CPU/PyTorch reference for unpack_seq."""
    import torch
    B, Lmax = packed.shape[:2]
    N = sum(lengths_list)
    out = torch.empty((N,) + packed.shape[2:], device=packed.device, dtype=packed.dtype)
    offset = 0
    for b in range(B):
        seq_len = lengths_list[b]
        out[offset:offset + seq_len] = packed[b, :seq_len]
        offset += seq_len
    return out
EXTRA_CASES = [{'packed_shape': (5, 17, 70), 'lengths': [0, 7, 8, 9, 17], 'dtype': 'float32', 'lengths_dtype': 'int64', 'block_t': 8, 'block_d': 32}, {'packed_shape': (3, 9, 2, 3, 5), 'lengths': [9, 0, 4], 'block_t': 16, 'block_d': 16}]

def run_control(index, load_module):
    if type(index) is not int or not 0 <= index < len(EXTRA_CASES):
        raise ValueError('Unknown upstream correctness control')
    import torch
    try:
        mod = load_module()
    except Exception as e:
        return (False, f'Failed to load module: {e}')
    device = 'cuda'
    for i, case in enumerate(CORRECTNESS_CASES):
        if i != index + 5:
            continue
        packed_shape = case['packed_shape']
        lengths_list = case['lengths']
        dtype = getattr(torch, case.get('dtype', 'float16'))
        lengths_dtype = getattr(torch, case.get('lengths_dtype', 'int32'))
        block_t = case.get('block_t', 64)
        block_d = case.get('block_d', 64)
        case_description = f'shape={packed_shape}, lengths={lengths_list}, dtype={dtype}, block_t={block_t}, block_d={block_d}'
        try:
            torch.manual_seed(42 + i)
            packed = torch.randn(*packed_shape, device=device, dtype=dtype)
            lengths = torch.tensor(lengths_list, device=device, dtype=lengths_dtype)
            result = mod.unpack_seq(packed, lengths, block_t=block_t, block_d=block_d)
            torch.cuda.synchronize()
            ref = reference_unpack_seq(packed, lengths_list)
            if result.shape != ref.shape:
                return (False, f'Case {i + 1} ({case_description}): output shape {tuple(result.shape)} != {tuple(ref.shape)}')
            if result.dtype != ref.dtype:
                return (False, f'Case {i + 1} ({case_description}): output dtype {result.dtype} != {ref.dtype}')
            if not torch.allclose(result, ref, atol=0.001, rtol=0.001):
                max_diff = (result - ref).abs().max().item()
                return (False, f'Case {i + 1} ({case_description}): max diff = {max_diff:.6f}')
        except Exception as e:
            return (False, f'Case {i + 1} ({case_description}): exception: {e}')
    return (True, None)
