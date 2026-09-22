"""Additional unscored public-branch controls from PR105.

Source revision: 0acf65b3a967ef1025dbfc5fd4b415b259e3bd43
Source path: tasks/triton2triton/vllm/triton_reduce_segments/scripts/task_runner.py
Original scored inputs, gates and timing remain in scripts/task_runner.py.
"""
import sys
import os
import json
import argparse
import importlib.util
TEST_SHAPES = [(4, 8, 64, 2, 128), (8, 16, 64, 4, 256), (16, 32, 128, 4, 512), (4, 8, 128, 2, 64), (32, 16, 64, 8, 1024)]
CORRECTNESS_CASES = [{'name': f'decode_{i + 1}', 'shape': shape} for i, shape in enumerate(TEST_SHAPES)] + [{'name': 'packed_variable_partial_fp16', 'shape': (4, 3, 80, 4, None), 'query_lens': (2, 0, 3, 1), 'seq_lens': (1, 127, 33, 65), 'segment_dtype': 'float16'}, {'name': 'zero_denom_extreme_max_float32_output', 'shape': (2, 2, 48, 4, None), 'query_lens': (1, 2), 'seq_lens': (64, 128), 'output_dtype': 'float32', 'special_values': 'zero_denom_extreme_max'}]

def make_test_data(num_seqs, num_query_heads, head_size, num_segments, seq_len_k, device='cuda', query_lens=None, seq_lens=None, segment_dtype='float32', output_dtype='float16'):
    import torch
    import triton
    head_size_padded = triton.next_power_of_2(head_size)
    if query_lens is None:
        query_lens = (1,) * num_seqs
    if len(query_lens) != num_seqs or any((length < 0 for length in query_lens)):
        raise ValueError('query_lens must contain one nonnegative length per sequence')
    total_tokens = sum(query_lens)
    segment_torch_dtype = getattr(torch, segment_dtype)
    output_torch_dtype = getattr(torch, output_dtype)
    torch.manual_seed(42)
    segm_output = torch.randn(total_tokens, num_query_heads, num_segments, head_size_padded, device=device, dtype=segment_torch_dtype)
    segm_max = torch.randn(total_tokens, num_query_heads, num_segments, device=device, dtype=segment_torch_dtype)
    segm_expsum = torch.rand(total_tokens, num_query_heads, num_segments, device=device, dtype=segment_torch_dtype) + 0.1
    output = torch.zeros(total_tokens, num_query_heads, head_size, device=device, dtype=output_torch_dtype)
    if seq_lens is None:
        seq_lens = (seq_len_k,) * num_seqs
    if len(seq_lens) != num_seqs or any((length <= 0 for length in seq_lens)):
        raise ValueError('seq_lens must contain one positive length per sequence')
    seqused_k = torch.tensor(seq_lens, device=device, dtype=torch.int32)
    cu_seqlens_q = torch.zeros(num_seqs + 1, device=device, dtype=torch.int32)
    cu_seqlens_q[1:] = torch.tensor(query_lens, device=device, dtype=torch.int32).cumsum(0)
    return (segm_output, segm_max, segm_expsum, output, seqused_k, cu_seqlens_q)

def reference_reduce(segm_output, segm_max, segm_expsum, head_size, seqused_k, cu_seqlens_q, tile_size=16):
    """PyTorch reference for logsumexp reduction."""
    import torch
    total_tokens = segm_output.shape[0]
    num_segments = segm_output.shape[2]
    token_indices = torch.arange(total_tokens, device=segm_output.device, dtype=torch.int32)
    seq_indices = torch.searchsorted(cu_seqlens_q, token_indices, right=True) - 1
    token_seq_lens = seqused_k[seq_indices]
    tiles_per_segment = torch.div(token_seq_lens + num_segments * tile_size - 1, num_segments * tile_size, rounding_mode='floor')
    active_segments = torch.div(token_seq_lens + tiles_per_segment * tile_size - 1, tiles_per_segment * tile_size, rounding_mode='floor')
    segment_mask = (torch.arange(num_segments, device=segm_output.device)[None, :] < active_segments[:, None])[:, None, :]
    segm_max_f32 = segm_max.float()
    segm_expsum_f32 = segm_expsum.float()
    segm_output_f32 = segm_output.float()
    masked_max = torch.where(segment_mask, segm_max_f32, -torch.inf)
    overall_max = masked_max.max(dim=-1).values
    max_scale = torch.exp(masked_max - overall_max.unsqueeze(-1))
    rescaled_expsum = torch.where(segment_mask, segm_expsum_f32 * max_scale, 0.0)
    overall_expsum = rescaled_expsum.sum(dim=-1)
    rescaled_output = torch.where(segment_mask.unsqueeze(-1), segm_output_f32 * max_scale.unsqueeze(-1), 0.0)
    summed = rescaled_output.sum(dim=2)
    denominator = overall_expsum.unsqueeze(-1)
    safe_denom = torch.where(denominator == 0, 1.0, denominator)
    output = torch.where(denominator == 0, 0.0, summed[:, :, :head_size] / safe_denom)
    return output
EXTRA_CASES = [{'name': 'packed_variable_partial_fp16', 'shape': (4, 3, 80, 4, None), 'query_lens': (2, 0, 3, 1), 'seq_lens': (1, 127, 33, 65), 'segment_dtype': 'float16'}, {'name': 'zero_denom_extreme_max_float32_output', 'shape': (2, 2, 48, 4, None), 'query_lens': (1, 2), 'seq_lens': (64, 128), 'output_dtype': 'float32', 'special_values': 'zero_denom_extreme_max'}]

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
        name = case['name']
        num_seqs, nqh, hs, nseg, slk = case['shape']
        try:
            torch.manual_seed(42 + i)
            segm_output, segm_max_t, segm_expsum, output, seqused_k, cu_seqlens_q = make_test_data(num_seqs, nqh, hs, nseg, slk, device, query_lens=case.get('query_lens'), seq_lens=case.get('seq_lens'), segment_dtype=case.get('segment_dtype', 'float32'), output_dtype=case.get('output_dtype', 'float16'))
            if case.get('special_values') == 'zero_denom_extreme_max':
                segm_expsum[0, 0, :] = 0
                segm_max_t[1, 0, :] = torch.tensor((-10000.0, -1000.0, 0.0, 10000.0), device=device, dtype=segm_max_t.dtype)
            mod.reduce_attention_segments(segm_output, segm_max_t, segm_expsum, output, seqused_k, cu_seqlens_q)
            torch.cuda.synchronize()
            ref = reference_reduce(segm_output, segm_max_t, segm_expsum, hs, seqused_k, cu_seqlens_q).to(output.dtype)
            if not torch.allclose(output.float(), ref.float(), atol=0.01, rtol=0.01):
                max_diff = (output.float() - ref.float()).abs().max().item()
                return (False, f'Case {name}: max diff = {max_diff:.6f}')
        except Exception as e:
            return (False, f'Case {name}: exception: {e}')
    return (True, None)
