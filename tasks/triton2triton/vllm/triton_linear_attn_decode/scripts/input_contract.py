"""Protected preconditions for one decode step; call outside device timing."""
import torch


INPUT_CONTRACT = {
    'token_count': 1,
    'qk_layout': 'B,H,1,D',
    'value_layout': 'B,H,1,D_v',
    'cache_layout': 'num_slots,H,D,D_v',
    'slope_layout': 'H',
    'slot_layout': 'B',
    'slot_dtypes': ['int32', 'int64'],
    'active_slots': 'unique_in_range',
    'padding_slot': -1,
    'padding_output': 'unwritten',
    'default_block_size': 32,
    'dimension_constraint': 'D_and_BLOCK_SIZE_power_of_two;D_v_divisible_by_BLOCK_SIZE',
}


def validate_inputs(q, k, v, kv_caches, slope_rate, slot_idx, BLOCK_SIZE=32):
    """Reject ambiguous slot updates before reference/candidate execution.

    Distinct active batch lanes own distinct cache slots. Repeated -1 padding is
    valid. Slot inspection may synchronize a device tensor, so this function is
    deliberately absent from the candidate wrapper and timed/captured callable.
    """
    tensors = (q, k, v, kv_caches, slope_rate, slot_idx)
    if not all(isinstance(value, torch.Tensor) for value in tensors):
        raise ValueError('Decode inputs must be tensors')
    if len({value.device for value in tensors}) != 1:
        raise ValueError('Decode inputs must share a device')
    if q.ndim != 4 or q.shape[2] != 1 or min(q.shape) <= 0:
        raise ValueError('q must have positive shape [B, H, 1, D]')
    batch, heads, _, dim = q.shape
    if k.shape != q.shape or v.ndim != 4 or v.shape[:3] != q.shape[:3] or v.shape[3] <= 0:
        raise ValueError('k/v must describe the same one-token batch and heads as q')
    value_dim = v.shape[3]
    if dim & (dim - 1):
        raise ValueError('The retained decode kernel requires a power-of-two D')
    if type(BLOCK_SIZE) is not int or BLOCK_SIZE <= 0 or BLOCK_SIZE & (BLOCK_SIZE - 1):
        raise ValueError('BLOCK_SIZE must be a positive power of two')
    if value_dim % BLOCK_SIZE:
        raise ValueError('The retained wrapper requires D_v divisible by BLOCK_SIZE')
    if (kv_caches.ndim != 4 or kv_caches.shape[0] <= 0
            or kv_caches.shape[1:] != (heads, dim, value_dim)):
        raise ValueError('kv_caches must have shape [num_slots, H, D, D_v]')
    if slope_rate.shape != (heads,):
        raise ValueError('slope_rate must have shape [H]')
    if slot_idx.shape != (batch,) or slot_idx.dtype not in (torch.int32, torch.int64):
        raise ValueError('slot_idx must be int32/int64 with shape [B]')
    slots = slot_idx.detach().cpu().tolist()
    active = [slot for slot in slots if slot != -1]
    if any(slot < 0 or slot >= kv_caches.shape[0] for slot in active):
        raise ValueError('Active slot indices must be in [0, num_slots); padding is -1')
    if len(active) != len(set(active)):
        raise ValueError('Active slot indices must be unique for one decode step')
