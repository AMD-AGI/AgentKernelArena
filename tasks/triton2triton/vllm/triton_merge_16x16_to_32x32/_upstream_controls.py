"""Unscored PR105 public-branch controls, ported from pinned main.

The original runner still owns every existing correctness and performance path.
These additional calls use the current checked candidate loader. No benchmark
helper or original workload is replaced.
Source revision: 0acf65b3a967ef1025dbfc5fd4b415b259e3bd43
Source path: tasks/triton2triton/vllm/triton_merge_16x16_to_32x32/scripts/task_runner.py
"""
import sys
import os
import json
import argparse
import importlib.util
SEEDS = [42, 43, 44, 45, 46]
CORRECTNESS_CASES = [{'seed': seed, 'B': 2, 'T': 64, 'H': 4, 'dtype': 'float32'} for seed in SEEDS] + [{'seed': 47, 'B': 1, 'T': 31, 'H': 1, 'dtype': 'float16'}, {'seed': 48, 'B': 3, 'T': 33, 'H': 2, 'dtype': 'bfloat16'}, {'seed': 49, 'B': 1, 'T': 47, 'H': 5, 'dtype': 'float32'}]

def reference(A):
    import torch
    B, T, H, BT = A.shape
    assert BT == 32
    from math import ceil
    NT = ceil(T / BT)
    Ai = torch.zeros_like(A, dtype=torch.float32, device='cpu')
    A_cpu = A.float().cpu()
    for b in range(B):
        for h in range(H):
            for t_idx in range(NT):
                start = t_idx * BT
                end = min(start + BT, T)
                sz = end - start
                block = A_cpu[b, start:end, h, :sz]
                I_plus_A = torch.eye(sz) + torch.tril(block, diagonal=-1)
                inv_block = torch.linalg.inv(I_plus_A)
                Ai[b, start:end, h, :sz] = inv_block
    return Ai

def gen_inputs(seed, device, B=2, T=64, H=4, dtype='float32'):
    import torch
    torch.manual_seed(seed)
    BT = 32
    A = torch.randn(B, T, H, BT, device=device, dtype=getattr(torch, dtype)) * 0.1
    idx = torch.arange(BT, device=device)
    t_in_block = torch.arange(T, device=device) % BT
    mask = t_in_block[:, None] > idx[None, :]
    A = A * mask[None, :, None, :]
    return ((A,), {})
EXTRA_CASES = [{'seed': 47, 'B': 1, 'T': 31, 'H': 1, 'dtype': 'float16'}, {'seed': 48, 'B': 3, 'T': 33, 'H': 2, 'dtype': 'bfloat16'}, {'seed': 49, 'B': 1, 'T': 47, 'H': 5, 'dtype': 'float32'}]

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
        try:
            args, kwargs = gen_inputs(device=device, **case)
            args_cpu = tuple((a.float().cpu() if isinstance(a, torch.Tensor) else a for a in args))
            result = mod.merge_16x16_to_32x32(*args, **kwargs)
            ref = reference(*args_cpu, **kwargs)
            r_cpu = result.float().cpu()
            ref_f = ref.float()
            if not torch.allclose(r_cpu, ref_f, atol=0.01, rtol=0.01):
                max_diff = (r_cpu - ref_f).abs().max().item()
                return (False, f'Case {i + 1} {case}: max diff = {max_diff:.6f}')
            torch.cuda.synchronize()
        except Exception as e:
            return (False, f'Case {i + 1} {case}: exception: {e}')
    return (True, None)
