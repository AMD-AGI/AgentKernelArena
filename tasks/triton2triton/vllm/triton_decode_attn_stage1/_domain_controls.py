"""Unscored input-domain controls for the protected ``decode_att_m_fwd`` wrapper.

These correctness-only cases span head dimensions, page sizes, split counts and
ragged sequence lengths beyond the five scored configurations. Each case uses a
seeded nonidentity page table, and every K/V slot that no active position
references holds a decoy value that must not affect the result. An independent
FP64 CPU oracle is computed from pristine clones before every candidate call.
Original scored inputs, tolerances, warmups, samples and timing remain in
scripts/task_runner.py; these controls add no performance rows.
"""
import math

import torch

SYMBOL = 'decode_att_m_fwd'
# The task's existing float16/bfloat16 numerical gate.
ATOL = 0.01
RTOL = 0.01
# V value in every K/V slot that no active position references.
DECOY_VALUE = 32.0
READONLY_SLOTS = (0, 1, 2, 4, 5)
TENSOR_SLOTS = (0, 1, 2, 3, 4, 5)

CASES = [
    {'name': 'page1_hd16_short_inactive', 'batch': 4, 'query_heads': 4, 'kv_heads': 4,
     'k_head_dim': 16, 'v_head_dim': 16, 'page_size': 1, 'num_splits': 4,
     'sequence_lengths': [1, 2, 5, 70], 'active_splits': [1, 2, 3, 4],
     'table_pages': 72, 'physical_pages': 81, 'dtype': 'float16', 'logit_cap': 0.0,
     'seeds': [2101, 2102], 'initial_partial_value': 23.5,
     'page_mapping': 'seeded permutation; decoy padding, pages and tail slots',
     'input_shapes': {'q': [4, 4, 16], 'k_buffer': [81, 4, 16], 'v_buffer': [81, 4, 16],
                      'req_to_tokens': [4, 72], 'b_seqlen': [4]},
     'output_shapes': {'att_out': [4, 4, 4, 17]}},
    {'name': 'page3_hd96_group3', 'batch': 3, 'query_heads': 6, 'kv_heads': 2,
     'k_head_dim': 96, 'v_head_dim': 96, 'page_size': 3, 'num_splits': 3,
     'sequence_lengths': [100, 7, 64], 'active_splits': [3, 3, 3],
     'table_pages': 36, 'physical_pages': 62, 'dtype': 'float16', 'logit_cap': 0.0,
     'seeds': [2201, 2202], 'initial_partial_value': 23.5,
     'page_mapping': 'seeded permutation; decoy padding, pages and tail slots',
     'input_shapes': {'q': [3, 6, 96], 'k_buffer': [186, 2, 96], 'v_buffer': [186, 2, 96],
                      'req_to_tokens': [3, 36], 'b_seqlen': [3]},
     'output_shapes': {'att_out': [3, 6, 3, 97]}},
    {'name': 'page48_hd32_mqa_one_split_capped', 'batch': 3, 'query_heads': 8, 'kv_heads': 1,
     'k_head_dim': 32, 'v_head_dim': 32, 'page_size': 48, 'num_splits': 1,
     'sequence_lengths': [129, 47, 48], 'active_splits': [1, 1, 1],
     'table_pages': 5, 'physical_pages': 8, 'dtype': 'float16', 'logit_cap': 1.0,
     'seeds': [2301, 2302], 'initial_partial_value': 23.5,
     'page_mapping': 'seeded permutation; decoy padding, pages and tail slots',
     'input_shapes': {'q': [3, 8, 32], 'k_buffer': [384, 1, 32], 'v_buffer': [384, 1, 32],
                      'req_to_tokens': [3, 5], 'b_seqlen': [3]},
     'output_shapes': {'att_out': [3, 8, 1, 33]}},
    {'name': 'page128_k192_v128', 'batch': 2, 'query_heads': 4, 'kv_heads': 2,
     'k_head_dim': 192, 'v_head_dim': 128, 'page_size': 128, 'num_splits': 2,
     'sequence_lengths': [300, 129], 'active_splits': [2, 2],
     'table_pages': 5, 'physical_pages': 8, 'dtype': 'float16', 'logit_cap': 0.0,
     'seeds': [2401, 2402], 'initial_partial_value': 23.5,
     'page_mapping': 'seeded permutation; decoy padding, pages and tail slots',
     'input_shapes': {'q': [2, 4, 192], 'k_buffer': [1024, 2, 192], 'v_buffer': [1024, 2, 128],
                      'req_to_tokens': [2, 5], 'b_seqlen': [2]},
     'output_shapes': {'att_out': [2, 4, 2, 129]}},
    {'name': 'page64_k80_v256', 'batch': 2, 'query_heads': 2, 'kv_heads': 1,
     'k_head_dim': 80, 'v_head_dim': 256, 'page_size': 64, 'num_splits': 4,
     'sequence_lengths': [65, 191], 'active_splits': [4, 4],
     'table_pages': 5, 'physical_pages': 8, 'dtype': 'float16', 'logit_cap': 0.0,
     'seeds': [2501, 2502], 'initial_partial_value': 23.5,
     'page_mapping': 'seeded permutation; decoy padding, pages and tail slots',
     'input_shapes': {'q': [2, 2, 80], 'k_buffer': [512, 1, 80], 'v_buffer': [512, 1, 256],
                      'req_to_tokens': [2, 5], 'b_seqlen': [2]},
     'output_shapes': {'att_out': [2, 2, 4, 257]}},
    {'name': 'long_page16_splits16', 'batch': 2, 'query_heads': 8, 'kv_heads': 2,
     'k_head_dim': 128, 'v_head_dim': 128, 'page_size': 16, 'num_splits': 16,
     'sequence_lengths': [7999, 4099], 'active_splits': [16, 16],
     'table_pages': 502, 'physical_pages': 760, 'dtype': 'float16', 'logit_cap': 0.0,
     'seeds': [2601, 2602], 'initial_partial_value': 23.5,
     'page_mapping': 'seeded permutation; decoy padding, pages and tail slots',
     'input_shapes': {'q': [2, 8, 128], 'k_buffer': [12160, 2, 128], 'v_buffer': [12160, 2, 128],
                      'req_to_tokens': [2, 502], 'b_seqlen': [2]},
     'output_shapes': {'att_out': [2, 8, 16, 129]}},
    {'name': 'bf16_page5_hd112_splits7_capped', 'batch': 3, 'query_heads': 4, 'kv_heads': 1,
     'k_head_dim': 112, 'v_head_dim': 112, 'page_size': 5, 'num_splits': 7,
     'sequence_lengths': [1, 33, 257], 'active_splits': [1, 7, 7],
     'table_pages': 54, 'physical_pages': 63, 'dtype': 'bfloat16', 'logit_cap': 2.5,
     'seeds': [2701, 2702], 'initial_partial_value': 23.5,
     'page_mapping': 'seeded permutation; decoy padding, pages and tail slots',
     'input_shapes': {'q': [3, 4, 112], 'k_buffer': [315, 1, 112], 'v_buffer': [315, 1, 112],
                      'req_to_tokens': [3, 54], 'b_seqlen': [3]},
     'output_shapes': {'att_out': [3, 4, 7, 113]}},
]


def _case(index):
    if type(index) is not int or not 0 <= index < len(CASES):
        raise ValueError('Unknown domain correctness control')
    return CASES[index]


def active_splits(lengths, num_splits):
    """Kernel split partition: ceil(length / splits) tokens per split; empty tails are inactive."""
    lengths = lengths.detach().cpu().long()
    chunk = (lengths + num_splits - 1) // num_splits
    return torch.arange(num_splits)[None, :] * chunk[:, None] < lengths[:, None]


def control_inputs(index, device='cuda', variant=0):
    """Protected generator. Variant 1 rotates lengths across requests and redraws every value."""
    case = _case(index)
    if variant not in (0, 1):
        raise ValueError('Unknown domain control variant')
    if case['dtype'] not in ('float16', 'bfloat16'):
        raise ValueError('Domain control data must be float16 or bfloat16')
    lengths = list(case['sequence_lengths'])
    if variant:
        lengths = lengths[-1:] + lengths[:-1]
    batch, heads, kv_heads = len(lengths), case['query_heads'], case['kv_heads']
    k_dim, v_dim = case['k_head_dim'], case['v_head_dim']
    page_size, splits = case['page_size'], case['num_splits']
    needed = [-(-length // page_size) for length in lengths]
    used, physical, width = sum(needed), case['physical_pages'], case['table_pages']
    decoys = physical - used
    if (batch != case['batch'] or min(lengths) < 1 or heads % kv_heads or decoys < 1
            or width <= max(needed) or not 16 <= min(k_dim, v_dim) <= max(k_dim, v_dim) <= 256):
        raise ValueError('Invalid protected domain control declaration')
    generator = torch.Generator().manual_seed(case['seeds'][variant])
    order = torch.randperm(physical, generator=generator)
    decoy_pages = order[used:]
    table = torch.empty(batch, width, dtype=torch.int64)
    offset = 0
    for request, count in enumerate(needed):
        table[request, :count] = order[offset:offset + count]
        offset += count
        # Padding entries point at decoy pages and must not affect the result.
        table[request, count:] = decoy_pages[(request + torch.arange(width - count)) % decoys]
    slots = physical * page_size
    q = torch.randn(batch, heads, k_dim, generator=generator)
    k = torch.randn(slots, kv_heads, k_dim, generator=generator)
    v = torch.randn(slots, kv_heads, v_dim, generator=generator)
    referenced = torch.zeros(slots, dtype=torch.bool)
    for request, length in enumerate(lengths):
        positions = torch.arange(length)
        referenced[table[request, positions // page_size] * page_size + positions % page_size] = True
    v[~referenced] = DECOY_VALUE
    b_seqlen = torch.tensor(lengths, dtype=torch.int32)
    active = active_splits(b_seqlen, splits)
    att_out = torch.full((batch, heads, splits, v_dim + 1), case['initial_partial_value'],
                         dtype=torch.float32)
    # Every slot the kernel must write starts poisoned; inactive slots keep caller state.
    att_out[active[:, None, :, None].expand_as(att_out)] = float('nan')
    shapes = {'q': list(q.shape), 'k_buffer': list(k.shape), 'v_buffer': list(v.shape),
              'req_to_tokens': list(table.shape), 'b_seqlen': list(b_seqlen.shape)}
    if (shapes != case['input_shapes'] or {'att_out': list(att_out.shape)} != case['output_shapes']
            or (variant == 0 and active.sum(1).tolist() != case['active_splits'])):
        raise ValueError('Domain control tensors differ from the protected manifest')
    dtype = getattr(torch, case['dtype'])
    args = (q.to(dtype).to(device), k.to(dtype).to(device), v.to(dtype).to(device),
            att_out.to(device), table.to(torch.int32).to(device), b_seqlen.to(device),
            splits, 1.0 / math.sqrt(k_dim), page_size)
    return args, {'logit_cap': float(case['logit_cap'])}


def reference(q, k, v, pages, lengths, num_splits, sm_scale, page_size, logit_cap=0.0):
    """Independent FP64 CPU oracle. Returns expected values and the active (request, split) mask."""
    q, k, v = (value.detach().cpu().double() for value in (q, k, v))
    pages, lengths = pages.detach().cpu().long(), lengths.detach().cpu().long()
    batch, heads, _ = q.shape
    kv_heads, v_dim = k.shape[1], v.shape[-1]
    if heads % kv_heads:
        raise ValueError('query heads must be a multiple of KV heads')
    head_to_kv = torch.arange(heads) // (heads // kv_heads)
    expected = torch.zeros(batch, heads, num_splits, v_dim + 1, dtype=torch.float64)
    active = active_splits(lengths, num_splits)
    for request in range(batch):
        length = int(lengths[request])
        chunk = -(-length // num_splits)
        positions = torch.arange(length)
        locations = pages[request, positions // page_size] * page_size + positions % page_size
        keys = k[locations][:, head_to_kv]
        values = v[locations][:, head_to_kv]
        logits = torch.einsum('hd,nhd->hn', q[request], keys) * sm_scale
        if logit_cap > 0:
            logits = logit_cap * torch.tanh(logits / logit_cap)
        for split in range(num_splits):
            begin, end = split * chunk, min(length, (split + 1) * chunk)
            if begin >= end:
                continue
            lse = torch.logsumexp(logits[:, begin:end], dim=1)
            weights = torch.exp(logits[:, begin:end] - lse[:, None])
            expected[request, :, split, :v_dim] = torch.einsum('hn,nhd->hd', weights, values[begin:end])
            expected[request, :, split, v_dim] = lse
    return expected, active


def _bytes(value):
    return value.detach().contiguous().reshape(-1).view(torch.uint8)


def _metadata(value):
    return value.shape, value.stride(), value.dtype, value.device, value.data_ptr()


def check_output(actual, expected, active, initial):
    """Every active element finite and within the gate; inactive caller state byte-exact."""
    if (not isinstance(actual, torch.Tensor) or actual.shape != initial.shape
            or actual.dtype != torch.float32):
        raise AssertionError('Domain control output shape/dtype mismatch')
    output = actual.detach().cpu()
    mask = active[:, None, :, None].expand(output.shape)
    written = output[mask]
    if not torch.isfinite(written).all():
        raise AssertionError('Domain control active partial output/logsumexp is unwritten or nonfinite')
    torch.testing.assert_close(written.double(), expected[mask], atol=ATOL, rtol=RTOL)
    if not torch.equal(_bytes(output[~mask]), _bytes(initial[~mask])):
        raise AssertionError('Domain control modified inactive caller-owned split slots')
    return int(mask.sum()), int((~mask).sum())


def checked_call(module, args, kwargs):
    tensors = {slot: args[slot] for slot in TENSOR_SLOTS}
    metadata = {slot: _metadata(value) for slot, value in tensors.items()}
    pristine = {slot: value.detach().clone() for slot, value in tensors.items()}
    initial = pristine[3].cpu()
    expected, active = reference(pristine[0], pristine[1], pristine[2], pristine[4],
                                 pristine[5], *args[6:], **kwargs)
    getattr(module, SYMBOL)(*args, **kwargs)
    if args[0].device.type == 'cuda':
        torch.cuda.synchronize()
    for slot, value in tensors.items():
        if _metadata(value) != metadata[slot]:
            raise AssertionError('Domain control changed caller tensor metadata/storage')
    for slot in READONLY_SLOTS:
        if not torch.equal(_bytes(tensors[slot]), _bytes(pristine[slot])):
            raise AssertionError('Domain control modified a read-only input')
    if args[3].device != args[0].device:
        raise AssertionError('Domain control output device mismatch')
    return check_output(args[3], expected, active, initial)


def run_control(index, load_module, device='cuda'):
    case = _case(index)
    try:
        module = load_module()
        args, kwargs = control_inputs(index, device, 0)
        first = checked_call(module, args, kwargs)
        # Same caller storage with changed contents: stale results cannot pass.
        changed, _ = control_inputs(index, device, 1)
        for slot in TENSOR_SLOTS:
            args[slot].copy_(changed[slot])
        second = checked_call(module, args, kwargs)
        return True, {'unscored_domain_control': True, 'independent_fp64_oracle_checked': True,
                      'readonly_inputs_checked': True, 'changed_input_second_call_checked': True,
                      'active_elements_checked': first[0] + second[0],
                      'inactive_elements_preserved': first[1] + second[1]}
    except Exception as exc:
        return False, f'Domain control {case["name"]}: {type(exc).__name__}: {exc}'


def validate_reference(harness):
    """CPU self-check for validate-task: original-oracle agreement and discrimination."""
    for index in range(len(CASES)):
        for variant in (0, 1):
            args, kwargs = control_inputs(index, 'cpu', variant)
            q, k, v, _, pages, lengths, splits, scale, page_size = args
            expected, active = reference(q, k, v, pages, lengths, splits, scale, page_size, **kwargs)
            mask = active[:, None, :, None].expand(expected.shape)
            wanted = expected[mask]
            if kwargs['logit_cap'] == 0:
                original = harness.reference_stage1(q, k, v, pages, lengths, splits, scale, page_size)
                torch.testing.assert_close(original.double()[mask], wanted, atol=1e-4, rtol=1e-4)
            else:
                uncapped, _ = reference(q, k, v, pages, lengths, splits, scale, page_size)
                if torch.allclose(uncapped[mask], wanted, atol=ATOL, rtol=RTOL):
                    raise AssertionError('Capped domain control does not distinguish uncapped logits')
            shifted, _ = reference(q, k, v, pages.roll(1, dims=1), lengths, splits, scale,
                                   page_size, **kwargs)
            for wrong in (torch.zeros_like(wanted), shifted[mask]):
                if torch.allclose(wrong, wanted, atol=ATOL, rtol=RTOL):
                    raise AssertionError('Domain control accepts zero output or wrong page routing')
