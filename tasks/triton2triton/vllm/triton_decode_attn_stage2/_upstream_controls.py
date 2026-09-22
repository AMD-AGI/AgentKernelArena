"""Additional unscored public-branch controls from PR105.

Source revision: 0acf65b3a967ef1025dbfc5fd4b415b259e3bd43
Source path: tasks/triton2triton/vllm/triton_decode_attn_stage2/scripts/task_runner.py
Original scored inputs, gates and timing remain in scripts/task_runner.py.
"""
import sys
import os
import json
import argparse
import importlib.util
TEST_SHAPES = [(1, 8, 8, 64, 128, 4, 16), (4, 16, 4, 64, 256, 8, 16), (2, 32, 8, 128, 512, 4, 32), (1, 8, 1, 64, 64, 2, 16), (8, 8, 8, 64, 128, 4, 16)]
CORRECTNESS_CASES = [(shape, None) for shape in TEST_SHAPES] + [((3, 6, 2, 80, 17, 3, 16), (17, 2, 11)), ((2, 4, 1, 64, 13, 1, 16), (13, 5))]

def reference_stage2(mid_o, b_seqlen, num_kv_splits, Lv):
    """
    CPU/PyTorch reference for decode attention stage2.

    Combines partial results from stage1 via logsumexp:
      max_lse = max(lse across splits)
      o = sum(exp(lse_i - max_lse) * mid_o_i) / sum(exp(lse_i - max_lse))
      final_lse = max_lse + log(sum(exp(lse_i - max_lse)))
    """
    import torch
    batch, num_heads = (mid_o.shape[0], mid_o.shape[1])
    head_dim = Lv
    o = torch.zeros(batch, num_heads, head_dim, device=mid_o.device, dtype=torch.float32)
    lse = torch.zeros(batch, num_heads, device=mid_o.device, dtype=torch.float32)
    for b in range(batch):
        seq_len = b_seqlen[b].item()
        kv_len_per_split = (seq_len + num_kv_splits - 1) // num_kv_splits
        for h in range(num_heads):
            e_max = -float('inf')
            e_sum = 0.0
            acc = torch.zeros(head_dim, device=mid_o.device, dtype=torch.float32)
            for s in range(num_kv_splits):
                start = kv_len_per_split * s
                end = min(start + kv_len_per_split, seq_len)
                if end <= start:
                    continue
                tv = mid_o[b, h, s, :Lv].float()
                tlogic = mid_o[b, h, s, Lv].float().item()
                n_e_max = max(tlogic, e_max)
                old_scale = torch.exp(torch.tensor(e_max - n_e_max))
                acc = acc * old_scale
                exp_logic = torch.exp(torch.tensor(tlogic - n_e_max))
                acc = acc + exp_logic * tv
                e_sum = e_sum * old_scale.item() + exp_logic.item()
                e_max = n_e_max
            if e_sum > 0:
                o[b, h, :] = acc / e_sum
                lse[b, h] = e_max + torch.log(torch.tensor(e_sum)).item()
    return (o, lse)

def make_stage1_outputs(bs, num_heads, num_kv_heads, head_dim, max_seq, num_kv_splits, page_size, device='cuda', dtype=None, seq_lengths=None):
    """
    Create synthetic stage1 outputs for testing stage2.

    We simulate what stage1 would produce: for each split, create a random
    partial attention output and a plausible logsumexp value.
    """
    import torch
    if dtype is None:
        dtype = torch.float16
    torch.manual_seed(42)
    Lv = head_dim
    mid_o = torch.zeros(bs, num_heads, num_kv_splits, Lv + 1, device=device, dtype=torch.float32)
    if seq_lengths is None:
        seq_lengths = (max_seq,) * bs
    if len(seq_lengths) != bs:
        raise ValueError('seq_lengths must contain one entry per batch element')
    if any((seq_len <= 0 or seq_len > max_seq for seq_len in seq_lengths)):
        raise ValueError('sequence lengths must be in the range [1, max_seq]')
    b_seqlen = torch.tensor(seq_lengths, device=device, dtype=torch.int32)
    for b in range(bs):
        seq_len = seq_lengths[b]
        kv_len_per_split = (seq_len + num_kv_splits - 1) // num_kv_splits
        for h in range(num_heads):
            for s in range(num_kv_splits):
                start = kv_len_per_split * s
                end = min(start + kv_len_per_split, seq_len)
                if end <= start:
                    continue
                mid_o[b, h, s, :Lv] = torch.randn(Lv, device=device)
                mid_o[b, h, s, Lv] = torch.randn(1, device=device).item() * 2.0
    o = torch.zeros(bs, num_heads, head_dim, device=device, dtype=dtype)
    lse = torch.zeros(bs, num_heads, device=device, dtype=torch.float32)
    v_buffer = torch.empty(1, num_kv_heads, head_dim, device=device, dtype=dtype)
    q = torch.empty(bs, num_heads, head_dim, device=device, dtype=dtype)
    return (mid_o, q, o, lse, v_buffer, b_seqlen)
EXTRA_CASES = [((3, 6, 2, 80, 17, 3, 16), (17, 2, 11)), ((2, 4, 1, 64, 13, 1, 16), (13, 5))]

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
    dtype = torch.float16
    for i, (shape, seq_lengths) in enumerate(EXTRA_CASES, start=5):
        if i != index + 5:
            continue
        bs, nh, nkv, hd, max_seq, num_splits, ps = shape
        try:
            mid_o, q, o, lse, v_buffer, b_seqlen = make_stage1_outputs(bs, nh, nkv, hd, max_seq, num_splits, ps, device, dtype, seq_lengths=seq_lengths)
            mod.decode_softmax_reducev_fwd(mid_o, q, o, lse, v_buffer, b_seqlen, num_splits)
            torch.cuda.synchronize()
            ref_o, ref_lse = reference_stage2(mid_o, b_seqlen, num_splits, hd)
            o_f32 = o.float()
            if not torch.allclose(o_f32, ref_o, atol=0.01, rtol=0.01):
                max_diff = (o_f32 - ref_o).abs().max().item()
                return (False, f'Shape {i + 1} output (bs={bs}, nh={nh}, hd={hd}, splits={num_splits}): max diff = {max_diff:.6f}')
            if not torch.allclose(lse, ref_lse, atol=0.01, rtol=0.01):
                max_diff = (lse - ref_lse).abs().max().item()
                return (False, f'Shape {i + 1} lse (bs={bs}, nh={nh}, hd={hd}, splits={num_splits}): max diff = {max_diff:.6f}')
        except Exception as e:
            return (False, f'Shape {i + 1} (bs={bs}, nh={nh}, nkv={nkv}, hd={hd}, seq={max_seq}, splits={num_splits}, ps={ps}): exception: {e}')
    return (True, None)
