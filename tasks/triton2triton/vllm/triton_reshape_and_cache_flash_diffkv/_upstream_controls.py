"""Unscored PR105 control inputs; original performance remains unchanged.
Source revision: 0acf65b3a967ef1025dbfc5fd4b415b259e3bd43
Source path: tasks/triton2triton/vllm/triton_reshape_and_cache_flash_diffkv/scripts/task_runner.py
"""
import sys
import os
import json
import argparse
import importlib.util

def reference_reshape_and_cache_diffkv(key, value, kv_cache, slot_mapping, kv_cache_dtype='auto', k_scale=None, v_scale=None):
    """CPU/PyTorch reference for reshape_and_cache_flash_diffkv."""
    import torch
    num_tokens = key.shape[0]
    num_heads = key.shape[1]
    head_size_k = key.shape[2]
    head_size_v = value.shape[2]
    block_size = kv_cache.shape[1]
    fp8_kv_cache = kv_cache_dtype != 'auto' and kv_cache_dtype.startswith('fp8')
    if fp8_kv_cache:
        fp8_dtypes = tuple((dtype for name in ('float8_e4m3fn', 'float8_e4m3fnuz', 'float8_e5m2', 'float8_e5m2fnuz') if (dtype := getattr(torch, name, None)) is not None))
        if key.dtype not in fp8_dtypes:
            key = key / k_scale
        if value.dtype not in fp8_dtypes:
            value = value / v_scale
    for i in range(num_tokens):
        slot = slot_mapping[i].item()
        if slot < 0:
            continue
        block_idx = slot // block_size
        block_offset = slot % block_size
        for h in range(num_heads):
            kv_cache[block_idx, block_offset, h, :head_size_k] = key[i, h]
            kv_cache[block_idx, block_offset, h, head_size_k:head_size_k + head_size_v] = value[i, h]
EXTRA_CASES = [{'name': 'padding_strided_bf16', 'shape': [7, 3, 17, 9, 2, 4], 'slots': [0, -1, 7, 3, -1, 4, 1]}, {'name': 'scaled_fp8', 'shape': [6, 2, 24, 40, 2, 4], 'k_scale': 0.75, 'v_scale': 1.25}]

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
    if index == 0:
        try:
            torch.manual_seed(100)
            num_tokens, num_heads, hk, hv, num_blocks, block_size = (7, 3, 17, 9, 2, 4)
            key = torch.randn(num_tokens * 2, num_heads, hk, device=device, dtype=torch.bfloat16)[::2]
            value = torch.randn(num_tokens * 2, num_heads, hv, device=device, dtype=torch.bfloat16)[::2]
            kv_cache_storage = torch.full((num_blocks * 2 + 1, block_size, num_heads, hk + hv), -2.0, device=device, dtype=torch.bfloat16)
            kv_cache = kv_cache_storage[1:1 + num_blocks * 2:2]
            kv_cache_ref_storage = kv_cache_storage.clone()
            kv_cache_ref = kv_cache_ref_storage[1:1 + num_blocks * 2:2]
            slot_mapping = torch.tensor([0, -1, num_blocks * block_size - 1, 3, -1, 4, 1], device=device, dtype=torch.int64)
            mod.reshape_and_cache_flash_diffkv(key, value, kv_cache, slot_mapping)
            torch.cuda.synchronize()
            reference_reshape_and_cache_diffkv(key, value, kv_cache_ref, slot_mapping)
            if not torch.equal(kv_cache_storage, kv_cache_ref_storage):
                max_diff = (kv_cache_storage.float() - kv_cache_ref_storage.float()).abs().max().item()
                return (False, f'Padding/strided BF16 case: kv_cache max diff = {max_diff:.6f}')
        except Exception as e:
            return (False, f'Padding/strided BF16 case: exception: {e}')
    if index == 1:
        try:
            torch.manual_seed(101)
            num_tokens, num_heads, hk, hv, num_blocks, block_size = (6, 2, 24, 40, 2, 4)
            key = torch.empty(num_tokens, num_heads, hk, device=device, dtype=dtype).uniform_(-3.0, 3.0)
            value = torch.empty(num_tokens, num_heads, hv, device=device, dtype=dtype).uniform_(-3.0, 3.0)
            fp8_dtype = torch.float8_e4m3fnuz if torch.version.hip else torch.float8_e4m3fn
            kv_cache = torch.full((num_blocks, block_size, num_heads, hk + hv), -1.0, device=device, dtype=fp8_dtype)
            kv_cache_ref = kv_cache.clone()
            slot_mapping = torch.tensor([num_blocks * block_size - 1, 0, 3, 2, 4, 1], device=device, dtype=torch.int64)
            k_scale = torch.tensor(0.75, device=device, dtype=torch.float32)
            v_scale = torch.tensor(1.25, device=device, dtype=torch.float32)
            mod.reshape_and_cache_flash_diffkv(key, value, kv_cache, slot_mapping, kv_cache_dtype='fp8', k_scale=k_scale, v_scale=v_scale)
            torch.cuda.synchronize()
            reference_reshape_and_cache_diffkv(key, value, kv_cache_ref, slot_mapping, kv_cache_dtype='fp8', k_scale=k_scale, v_scale=v_scale)
            if not torch.equal(kv_cache.float(), kv_cache_ref.float()):
                max_diff = (kv_cache.float() - kv_cache_ref.float()).abs().max().item()
                return (False, f'Scaled FP8 case: kv_cache max diff = {max_diff:.6f}')
        except Exception as e:
            return (False, f'Scaled FP8 case: exception: {e}')
    return (True, None)
