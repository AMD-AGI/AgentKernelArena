"""Additional unscored public-branch controls from PR105.

Source revision: 0acf65b3a967ef1025dbfc5fd4b415b259e3bd43
Source path: tasks/triton2triton/vllm/triton_decode_attn_stage1/scripts/task_runner.py
Original scored inputs, gates and timing remain in scripts/task_runner.py.
"""
import sys
import os
import json
import argparse
import importlib.util
TEST_SHAPES = [(1, 8, 8, 64, 128, 4, 16), (4, 16, 4, 64, 256, 8, 16), (2, 32, 8, 128, 512, 4, 32), (1, 8, 1, 64, 64, 2, 16), (8, 8, 8, 64, 128, 4, 16)]
CORRECTNESS_CASES = [{'name': f'performance_shape_{i + 1}', 'shape': shape, 'sequence_lengths': None, 'dtype': 'float16', 'logit_cap': 0.0} for i, shape in enumerate(TEST_SHAPES)] + [{'name': 'ragged_gqa_empty_splits', 'shape': (3, 12, 3, 80, 197, 5, 8), 'sequence_lengths': (1, 130, 197), 'dtype': 'float16', 'logit_cap': 0.0}, {'name': 'ragged_capped_bfloat16_mqa', 'shape': (2, 8, 1, 256, 257, 3, 64), 'sequence_lengths': (65, 257), 'dtype': 'bfloat16', 'logit_cap': 1.5}]

def reference_stage1(q, k_buffer, v_buffer, req_to_tokens, b_seqlen, num_kv_splits, sm_scale, page_size, logit_cap=0.0):
    """
    CPU/PyTorch reference for decode attention stage1.

    For each (batch, head, kv_split), compute partial attention over
    the assigned KV range and return partial output + logsumexp.
    """
    import torch
    batch, num_heads, head_dim = q.shape
    num_kv_heads = k_buffer.shape[1]
    kv_group_num = num_heads // num_kv_heads
    Lv = v_buffer.shape[-1]
    att_out = torch.zeros(batch, num_heads, num_kv_splits, Lv + 1, device=q.device, dtype=torch.float32)
    for b in range(batch):
        seq_len = b_seqlen[b].item()
        kv_len_per_split = (seq_len + num_kv_splits - 1) // num_kv_splits
        for h in range(num_heads):
            kv_h = h // kv_group_num
            q_vec = q[b, h, :].float()
            for s in range(num_kv_splits):
                start = kv_len_per_split * s
                end = min(start + kv_len_per_split, seq_len)
                if end <= start:
                    continue
                positions = torch.arange(start, end, device=q.device)
                page_nums = req_to_tokens[b, positions // page_size]
                kv_locs = page_nums * page_size + positions % page_size
                k_vals = k_buffer[kv_locs, kv_h, :].float()
                v_vals = v_buffer[kv_locs, kv_h, :].float()
                scores = k_vals @ q_vec * sm_scale
                if logit_cap > 0:
                    scores = logit_cap * torch.tanh(scores / logit_cap)
                max_score = scores.max()
                exp_scores = torch.exp(scores - max_score)
                sum_exp = exp_scores.sum()
                partial_out = (exp_scores.unsqueeze(-1) * v_vals).sum(0) / sum_exp
                att_out[b, h, s, :Lv] = partial_out
                att_out[b, h, s, Lv] = max_score + torch.log(sum_exp)
    return att_out

def make_inputs(bs, num_heads, num_kv_heads, head_dim, max_seq, num_kv_splits, page_size, device='cuda', dtype=None, sequence_lengths=None, shuffled_page_table=True):
    """Create test inputs for the stage1 kernel."""
    import torch
    if dtype is None:
        dtype = torch.float16
    if sequence_lengths is None:
        sequence_lengths = (max_seq,) * bs
    if len(sequence_lengths) != bs:
        raise ValueError('sequence_lengths must contain one entry per batch')
    if any((seq_len <= 0 or seq_len > max_seq for seq_len in sequence_lengths)):
        raise ValueError('sequence lengths must be in the range [1, max_seq]')
    torch.manual_seed(42)
    q = torch.randn(bs, num_heads, head_dim, device=device, dtype=dtype)
    max_pages_per_seq = (max_seq + page_size - 1) // page_size
    if shuffled_page_table:
        physical_pages_per_seq = 2 * max_pages_per_seq + 1
        total_pages = bs * physical_pages_per_seq
    else:
        total_pages = bs * max_pages_per_seq
    total_tokens = total_pages * page_size
    k_buffer = torch.randn(total_tokens, num_kv_heads, head_dim, device=device, dtype=dtype)
    v_buffer = torch.randn(total_tokens, num_kv_heads, head_dim, device=device, dtype=dtype)
    max_seq_padded = max_pages_per_seq * page_size
    req_to_tokens = torch.zeros(bs, max_seq_padded, device=device, dtype=torch.int32)
    if shuffled_page_table:
        logical_pages = torch.arange(max_pages_per_seq - 1, -1, -1, device=device, dtype=torch.int32)
        for b in range(bs):
            page_order = torch.roll(logical_pages, shifts=b)
            page_base = b * physical_pages_per_seq
            req_to_tokens[b, :max_pages_per_seq] = page_base + 1 + 2 * page_order
    else:
        for b in range(bs):
            for pos in range(max_seq):
                page_idx = b * max_pages_per_seq + pos // page_size
                req_to_tokens[b, pos] = page_idx
    b_seqlen = torch.tensor(sequence_lengths, device=device, dtype=torch.int32)
    att_out = torch.zeros(bs, num_heads, num_kv_splits, head_dim + 1, device=device, dtype=torch.float32)
    sm_scale = 1.0 / head_dim ** 0.5
    return (q, k_buffer, v_buffer, att_out, req_to_tokens, b_seqlen, sm_scale)
EXTRA_CASES = [{'name': 'ragged_gqa_empty_splits', 'shape': (3, 12, 3, 80, 197, 5, 8), 'sequence_lengths': (1, 130, 197), 'dtype': 'float16', 'logit_cap': 0.0}, {'name': 'ragged_capped_bfloat16_mqa', 'shape': (2, 8, 1, 256, 257, 3, 64), 'sequence_lengths': (65, 257), 'dtype': 'bfloat16', 'logit_cap': 1.5}]

def run_control(index, load_module):
    if type(index) is not int or not 0 <= index < len(EXTRA_CASES):
        raise ValueError('Unknown upstream correctness control')
    'Run correctness checks against PyTorch reference.'
    import torch
    try:
        mod = load_module()
    except Exception as e:
        return (False, f'Failed to load module: {e}')
    device = 'cuda'
    for i, case in enumerate(EXTRA_CASES, start=5):
        if i != index + 5:
            continue
        bs, nh, nkv, hd, max_seq, num_splits, ps = case['shape']
        dtype = getattr(torch, case['dtype'])
        logit_cap = case['logit_cap']
        try:
            q, k_buf, v_buf, att_out, req_to_tokens, b_seqlen, sm_scale = make_inputs(bs, nh, nkv, hd, max_seq, num_splits, ps, device, dtype, sequence_lengths=case['sequence_lengths'], shuffled_page_table=True)
            mod.decode_att_m_fwd(q, k_buf, v_buf, att_out, req_to_tokens, b_seqlen, num_splits, sm_scale, ps, logit_cap=logit_cap)
            torch.cuda.synchronize()
            ref = reference_stage1(q, k_buf, v_buf, req_to_tokens, b_seqlen, num_splits, sm_scale, ps, logit_cap)
            if not torch.allclose(att_out, ref, atol=0.01, rtol=0.01):
                max_diff = (att_out - ref).abs().max().item()
                return (False, f'Case {i + 1} {case['name']} (bs={bs}, nh={nh}, nkv={nkv}, hd={hd}, max_seq={max_seq}, b_seqlen={tuple(b_seqlen.tolist())}, splits={num_splits}, ps={ps}, dtype={case['dtype']}, logit_cap={logit_cap}): max diff = {max_diff:.6f}')
        except Exception as e:
            return (False, f'Case {i + 1} {case['name']} (bs={bs}, nh={nh}, nkv={nkv}, hd={hd}, max_seq={max_seq}, splits={num_splits}, ps={ps}, dtype={case['dtype']}, logit_cap={logit_cap}): exception: {e}')
    return (True, None)
