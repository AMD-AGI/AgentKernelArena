"""Unscored controls for expert-bit packing.

The first three controls come from PR105 (revision
0acf65b3a967ef1025dbfc5fd4b415b259e3bd43, task runner path
tasks/triton2triton/vllm/triton_pack_bitmatrix/scripts/task_runner.py).
The final two top-k boundary controls and three strided-input controls were
added in this repository.
Original scored inputs, gates and timing remain in scripts/task_runner.py.
"""
import sys, os, json, argparse, importlib.util

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
    if topk > 32:
        # Each row has a distinct first-tile signature and a duplicated ID.
        # Later tiles add bits absent from that row's preceding tiles, so
        # skipping a tile or reusing row 0's later assignments is observable.
        first = [0, 31, 32, 63, min(64, num_experts - 1)]
        middle = [33, 34, 35, 36, 37]
        final = [num_experts - 1 - i for i in range(5)]
        for row in range(n_rows):
            slot = row % 5
            topk_ids[row, :32] = first[slot]
            topk_ids[row, 1] = first[slot]  # An explicit duplicate.
            topk_ids[row, 31] = slot + 1
            if topk > 33:
                topk_ids[row, 32:64] = middle[slot]
            topk_ids[row, -1] = final[slot]
    return topk_ids.to(device)


def make_strided_topk_ids(n_rows, num_experts, topk, layout, device):
    """Keep the physical gaps on the target device for layout controls."""
    import torch
    strides = {'row_padding': (4, 1), 'column_gaps': (4, 2),
               'row_and_column_gaps': (7, 3)}
    if (n_rows, num_experts, topk) != (3, 34, 2) or layout not in strides:
        raise ValueError('Unknown strided bitmatrix control')
    row_stride, col_stride = strides[layout]
    storage_size = (n_rows - 1) * row_stride + (topk - 1) * col_stride + 1
    storage = torch.zeros(storage_size, dtype=torch.int16, device=device)
    ids = storage.as_strided((n_rows, topk), (row_stride, col_stride))
    ids.copy_(torch.tensor([[0, 31], [1, 32], [2, 33]],
                           dtype=torch.int16, device=device))
    assert ids.stride() == (row_stride, col_stride) and not ids.is_contiguous()
    return ids


EXTRA_CASES = [(17, 31, 1), (33, 32, 31), (513, 33, 32),
               (5, 70, 33), (5, 70, 65),
               (3, 34, 2, 'row_padding'),
               (3, 34, 2, 'column_gaps'),
               (3, 34, 2, 'row_and_column_gaps')]

def run_control(index, load_module):
    if type(index) is not int or not 0 <= index < len(EXTRA_CASES):
        raise ValueError('Unknown upstream correctness control')
    import torch
    try:
        mod = load_module()
    except Exception as e:
        return (False, f'Failed to load module: {e}')
    device = 'cuda'
    for i, control in enumerate(EXTRA_CASES, start=0):
        if i != index + 0:
            continue
        try:
            n_rows, num_experts, topk = control[:3]
            topk_ids = (make_strided_topk_ids(*control, device)
                        if len(control) == 4 else
                        make_boundary_topk_ids(n_rows, num_experts, topk, device))
            result = mod.pack_topk_to_bitmatrix(topk_ids, num_experts)
            torch.cuda.synchronize()
            ref = reference_pack_bitmatrix(topk_ids.cpu(), num_experts).to(device)
            if not torch.equal(result, ref):
                diff_count = (result != ref).sum().item()
                return (False, f'Boundary shape {i + 1}: {diff_count} mismatched elements')
        except Exception as e:
            return (False, f'Boundary shape {i + 1}: exception: {e}')
    return (True, None)
