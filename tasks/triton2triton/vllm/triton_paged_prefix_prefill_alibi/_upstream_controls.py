"""Additional unscored public-branch controls from PR105.

Source revision: 0acf65b3a967ef1025dbfc5fd4b415b259e3bd43
Source path: tasks/triton2triton/vllm/triton_paged_prefix_prefill_alibi/scripts/task_runner.py
Original scored inputs, gates and timing remain in scripts/task_runner.py.
"""
import sys
import os
import json
import argparse
import time
import importlib.util
import math
ADDITIONAL_CORRECTNESS_CASES = [{'name': 'ragged_partial_shuffled_explicit_scale', 'context_lens': (17, 32, 47), 'query_lens': (7, 129, 33), 'num_heads': 8, 'num_kv_heads': 2, 'head_dim': 64, 'block_size': 16, 'sm_scale': 0.2}]

def get_alibi_slopes(num_heads):
    """Generate ALiBi slopes for the given number of heads."""
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = 2 ** (-2 ** (-(math.log2(closest_power_of_2) - 3)))
    powers = [base ** (i + 1) for i in range(closest_power_of_2)]
    if closest_power_of_2 != num_heads:
        extra_base = 2 ** (-2 ** (-(math.log2(2 * closest_power_of_2) - 3)))
        extra_powers = [extra_base ** (2 * i + 1) for i in range(num_heads - closest_power_of_2)]
        powers = powers + extra_powers
    return powers[:num_heads]

def setup_ragged_paged_kv_cache(context_lens, num_kv_heads, head_dim, block_size, device, dtype):
    """Create a ragged cache with deterministic, gapped physical block IDs."""
    import torch
    batch_size = len(context_lens)
    x = 8
    assert head_dim % x == 0
    blocks_per_seq = [(ctx_len + block_size - 1) // block_size for ctx_len in context_lens]
    num_mapped_blocks = sum(blocks_per_seq)
    max_blocks_per_seq = max(blocks_per_seq)
    physical_block_ids = list(range(1, 2 * num_mapped_blocks, 2))
    split = (num_mapped_blocks + 1) // 2
    physical_block_ids = physical_block_ids[split:] + physical_block_ids[:split]
    total_blocks = 2 * num_mapped_blocks + 2
    k_cache = torch.zeros(total_blocks, num_kv_heads, head_dim // x, block_size, x, device=device, dtype=dtype)
    v_cache = torch.zeros(total_blocks, num_kv_heads, head_dim, block_size, device=device, dtype=dtype)
    b_loc = torch.zeros(batch_size, max_blocks_per_seq, device=device, dtype=torch.int32)
    max_context_len = max(context_lens)
    full_k_ctx = torch.zeros(batch_size, max_context_len, num_kv_heads, head_dim, device=device, dtype=dtype)
    full_v_ctx = torch.zeros_like(full_k_ctx)
    mapping_idx = 0
    for batch_idx, ctx_len in enumerate(context_lens):
        full_k_ctx[batch_idx, :ctx_len] = torch.randn(ctx_len, num_kv_heads, head_dim, device=device, dtype=dtype)
        full_v_ctx[batch_idx, :ctx_len] = torch.randn(ctx_len, num_kv_heads, head_dim, device=device, dtype=dtype)
        for logical_block in range(blocks_per_seq[batch_idx]):
            physical_block = physical_block_ids[mapping_idx]
            mapping_idx += 1
            b_loc[batch_idx, logical_block] = physical_block
            start_pos = logical_block * block_size
            end_pos = min(start_pos + block_size, ctx_len)
            length = end_pos - start_pos
            k_values = full_k_ctx[batch_idx, start_pos:end_pos]
            v_values = full_v_ctx[batch_idx, start_pos:end_pos]
            k_cache[physical_block, :, :, :length, :] = k_values.permute(1, 2, 0).reshape(num_kv_heads, head_dim // x, x, length).permute(0, 1, 3, 2)
            v_cache[physical_block, :, :, :length] = v_values.permute(1, 2, 0)
    return (k_cache, v_cache, b_loc, full_k_ctx, full_v_ctx)

def reference_attention_alibi(q_packed, k_new_packed, v_new_packed, full_k_ctx, full_v_ctx, b_start_loc, b_seq_len, alibi_slopes, batch_size, ctx_len, query_len, num_heads, num_kv_heads, head_dim, sm_scale=None):
    """
    CPU/PyTorch reference for paged prefix prefill attention with ALiBi.
    """
    import torch
    kv_group_num = num_heads // num_kv_heads
    out = torch.zeros_like(q_packed)
    if sm_scale is None:
        sm_scale = 1.0 / head_dim ** 0.5
    context_lens = [ctx_len] * batch_size if isinstance(ctx_len, int) else list(ctx_len)
    query_lens = [query_len] * batch_size if isinstance(query_len, int) else list(query_len)
    for b in range(batch_size):
        start = b_start_loc[b].item()
        total_len = b_seq_len[b].item()
        ctx_len_b = context_lens[b]
        q_len = query_lens[b]
        assert total_len == ctx_len_b + q_len
        for h in range(num_heads):
            kv_h = h // kv_group_num
            slope = alibi_slopes[h].item()
            q_b = q_packed[start:start + q_len, h, :]
            k_ctx = full_k_ctx[b, :ctx_len_b, kv_h, :]
            v_ctx = full_v_ctx[b, :ctx_len_b, kv_h, :]
            k_new = k_new_packed[start:start + q_len, kv_h, :]
            v_new = v_new_packed[start:start + q_len, kv_h, :]
            k_full = torch.cat([k_ctx, k_new], dim=0)
            v_full = torch.cat([v_ctx, v_new], dim=0)
            scores = q_b @ k_full.T * sm_scale
            S = scores.shape[1]
            q_positions = torch.arange(ctx_len_b, ctx_len_b + q_len, device=scores.device).float()
            k_positions = torch.arange(0, S, device=scores.device).float()
            alibi_bias = slope * (k_positions[None, :] - q_positions[:, None])
            alibi_bias = torch.where(alibi_bias <= 0, alibi_bias, torch.tensor(float('-inf')))
            for qi in range(q_len):
                for ki in range(q_len):
                    if ki > qi:
                        scores[qi, ctx_len_b + ki] = float('-inf')
                        alibi_bias[qi, ctx_len_b + ki] = float('-inf')
            scores = scores + alibi_bias
            attn = torch.softmax(scores.float(), dim=-1).to(q_b.dtype)
            out[start:start + q_len, h, :] = attn @ v_full
    return out
EXTRA_CASES = [{'name': 'ragged_partial_shuffled_explicit_scale', 'context_lens': (17, 32, 47), 'query_lens': (7, 129, 33), 'num_heads': 8, 'num_kv_heads': 2, 'head_dim': 64, 'block_size': 16, 'sm_scale': 0.2}]

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
        name = case['name']
        context_lens = case['context_lens']
        query_lens = case['query_lens']
        nh = case['num_heads']
        nkv = case['num_kv_heads']
        hd = case['head_dim']
        blk_sz = case['block_size']
        sm_scale = case['sm_scale']
        bs = len(context_lens)
        try:
            assert len(query_lens) == bs
            torch.manual_seed(1000 + i)
            total_tokens = sum(query_lens)
            q = torch.randn(total_tokens, nh, hd, device=device, dtype=dtype)
            k_new = torch.randn(total_tokens, nkv, hd, device=device, dtype=dtype)
            v_new = torch.randn(total_tokens, nkv, hd, device=device, dtype=dtype)
            o = torch.zeros_like(q)
            k_cache, v_cache, b_loc, full_k_ctx, full_v_ctx = setup_ragged_paged_kv_cache(context_lens, nkv, hd, blk_sz, device, dtype)
            b_start_loc = torch.zeros(bs + 1, device=device, dtype=torch.int32)
            b_start_loc[1:] = torch.tensor(query_lens, device=device, dtype=torch.int32).cumsum(0)
            b_seq_len = torch.tensor([ctx + query for ctx, query in zip(context_lens, query_lens)], device=device, dtype=torch.int32)
            slopes = get_alibi_slopes(nh)
            alibi_slopes = torch.tensor(slopes, device=device, dtype=torch.float32)
            mod.context_attention_fwd_alibi(q, k_new, v_new, o, k_cache, v_cache, b_loc, b_start_loc, b_seq_len, max_input_len=max(query_lens), alibi_slopes=alibi_slopes, sm_scale=sm_scale)
            torch.cuda.synchronize()
            ref = reference_attention_alibi(q, k_new, v_new, full_k_ctx, full_v_ctx, b_start_loc, b_seq_len, alibi_slopes, bs, context_lens, query_lens, nh, nkv, hd, sm_scale=sm_scale)
            if not torch.allclose(o, ref, atol=0.01, rtol=0.01):
                max_diff = (o - ref).abs().max().item()
                return (False, f'Case {name}: max diff = {max_diff:.6f}')
        except Exception as e:
            return (False, f'Case {name}: exception: {e}')
    return (True, None)
