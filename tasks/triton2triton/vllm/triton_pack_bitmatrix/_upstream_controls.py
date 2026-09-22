"""Additional unscored public-branch controls from PR105.

Source revision: 0acf65b3a967ef1025dbfc5fd4b415b259e3bd43
Source path: tasks/triton2triton/vllm/triton_pack_bitmatrix/scripts/task_runner.py
Original scored inputs, gates and timing remain in scripts/task_runner.py.
"""
import sys, os, json, argparse, importlib.util
BOUNDARY_TEST_SHAPES = [(17, 31, 1), (33, 32, 31), (513, 33, 32)]

def reference_pack_bitmatrix(topk_ids, num_experts):
    """Independent CPU expert-membership oracle; padding is not an assignment."""
    import torch
    n_rows, topk = topk_ids.shape
    bitmatrix = torch.zeros(n_rows, (num_experts + 31) // 32, dtype=torch.uint32)
    for row in range(n_rows):
        for eid in topk_ids[row].tolist():
            if not 0 <= eid < num_experts:
                raise ValueError("Expert ID is outside the declared expert range")
            col, bit = divmod(eid, 32)
            bitmatrix[row, col] = bitmatrix[row, col].item() | (1 << bit)
    return bitmatrix

def make_boundary_topk_ids(n_rows, num_experts, topk, device):
    """Build deterministic valid IDs with duplicates and word-edge values."""
    import torch
    rows = torch.arange(n_rows, dtype=torch.int64)[:, None]
    cols = torch.arange(topk, dtype=torch.int64)[None, :]
    topk_ids = ((rows * 17 + cols * 7) % num_experts).to(torch.int16)
    edge_ids = [0, 0, min(30, num_experts - 1), min(31, num_experts - 1), min(32, num_experts - 1), num_experts - 1, num_experts - 1]
    prefix_len = min(topk, len(edge_ids))
    topk_ids[0, :prefix_len] = torch.tensor(edge_ids[:prefix_len], dtype=torch.int16)
    topk_ids[-1, 0] = num_experts - 1
    return topk_ids.to(device)
EXTRA_CASES = [(17, 31, 1), (33, 32, 31), (513, 33, 32)]

def run_control(index, load_module):
    if type(index) is not int or not 0 <= index < len(EXTRA_CASES):
        raise ValueError('Unknown upstream correctness control')
    import torch
    try:
        mod = load_module()
    except Exception as e:
        return (False, f'Failed to load module: {e}')
    device = 'cuda'
    for i, (n_rows, num_experts, topk) in enumerate(EXTRA_CASES, start=0):
        if i != index + 0:
            continue
        try:
            topk_ids = make_boundary_topk_ids(n_rows, num_experts, topk, device)
            result = mod.pack_topk_to_bitmatrix(topk_ids, num_experts)
            torch.cuda.synchronize()
            ref = reference_pack_bitmatrix(topk_ids.cpu(), num_experts).to(device)
            if not torch.equal(result, ref):
                diff_count = (result != ref).sum().item()
                return (False, f'Boundary shape {i + 1}: {diff_count} mismatched elements')
        except Exception as e:
            return (False, f'Boundary shape {i + 1}: exception: {e}')
    return (True, None)
