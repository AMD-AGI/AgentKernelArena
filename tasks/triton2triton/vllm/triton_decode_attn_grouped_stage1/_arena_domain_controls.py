"""Protected correctness-only controls spanning the public wrapper's input domain.

The protected wrapper ``decode_grouped_att_m_fwd`` fixes BLOCK_N=32, BLOCK_H=16,
BLOCK_DPE=0, num_warps=4 and num_stages=2. It derives everything else from its
arguments: Lk/Lv from the K/V buffers (each padded to one power-of-two tile and
masked), kv_group_num = q_heads // kv_heads, and the PAGE_SIZE / NUM_KV_SPLITS
compile-time values. These cases vary exactly those inputs beyond the scored
shapes. They are unscored: ``_arena_checks`` runs them through the same public
wrapper, launch audit, poisoning, read-only, inactive-slot and graph-replay checks
as the other controls, but compares against the independent float64 oracle below.
"""

_COMMON = {'dtypes': {'data': 'float16', 'partial_or_statistics': 'float32', 'routing': 'int32'},
           'page_mapping': 'seeded permutation of physical pages',
           'initial_partial_value': 23.5}

# (name, batch, query_heads, kv_heads, Lk, Lv, page_size, num_splits, lengths, logit_cap, seed)
_SPECS = (
    ('domain_group1_page1', 2, 3, 3, 64, 64, 1, 3, (45, 7), 0.0, 901),
    ('domain_group3_page5_dim80', 2, 6, 2, 80, 80, 5, 2, (53, 21), 1.5, 902),
    ('domain_group5_page3_k96_v64', 2, 15, 3, 96, 64, 3, 5, (70, 3), 0.0, 903),
    ('domain_group7_page64_dim256', 2, 14, 2, 256, 256, 64, 1, (100, 33), 0.0, 904),
    ('domain_group33_page12_k40_v72', 2, 66, 2, 40, 72, 12, 6, (77, 1), 2.5, 905),
    ('domain_mqa_group12_page2_dim16', 3, 12, 1, 16, 16, 2, 4, (31, 64, 2), 0.0, 906),
    ('domain_group16_page16_dim128', 2, 32, 2, 128, 128, 16, 7, (130, 40), 1.5, 907),
)


def _case(name, batch, heads, kv_heads, lk, lv, page_size, splits, lengths, cap, seed):
    if heads % kv_heads or len(lengths) != batch or min(lengths) < 1:
        raise ValueError(f'Invalid domain control {name}')
    max_seq = max(lengths)
    # One spare logical column per request keeps every table entry a valid
    # physical page after the replay check rolls each row by one column.
    columns = -(-max_seq // page_size) + 1
    physical = batch * columns + 1  # One physical page is never referenced.
    return {'contract_case': name, 'batch': batch, 'query_heads': heads, 'kv_heads': kv_heads,
            'kv_group_num': heads // kv_heads, 'head_dim': lk, 'value_head_dim': lv,
            'max_seq': max_seq, 'num_splits': splits, 'page_size': page_size,
            'table_pages_per_request': columns, 'physical_pages': physical,
            'sequence_lengths': list(lengths), 'page_mapping': _COMMON['page_mapping'],
            'logit_cap': cap,
            'input_shapes': {'q': [batch, heads, lk],
                             'k_buffer': [physical * page_size, kv_heads, lk],
                             'v_buffer': [physical * page_size, kv_heads, lv],
                             'req_to_tokens': [batch, columns],
                             'b_seqlen': [batch]},
            'output_shapes': {'att_out': [batch, heads, splits, lv + 1]},
            'dtypes': dict(_COMMON['dtypes']), 'seed': seed,
            'initial_partial_value': _COMMON['initial_partial_value']}


DOMAIN_CASES = {spec[0]: _case(*spec) for spec in _SPECS}


def control_inputs(case, device='cuda'):
    """Draw on a seeded CPU generator so CPU tests and GPU runs see equal bytes."""
    import torch
    c = DOMAIN_CASES[case]
    shapes = c['input_shapes']
    generator = torch.Generator(device='cpu').manual_seed(c['seed'])

    def normal(shape):
        return torch.randn(shape, generator=generator, dtype=torch.float32).to(
            device=device, dtype=torch.float16)

    q, k, v = normal(shapes['q']), normal(shapes['k_buffer']), normal(shapes['v_buffer'])
    batch, columns = shapes['req_to_tokens']
    order = torch.randperm(c['physical_pages'], generator=generator)[:batch * columns]
    pages = order.reshape(batch, columns).to(device=device, dtype=torch.int32)
    lengths = torch.tensor(c['sequence_lengths'], device=device, dtype=torch.int32)
    out = torch.full(c['output_shapes']['att_out'], c['initial_partial_value'],
                     device=device, dtype=torch.float32)
    scale = 1.0 / c['head_dim'] ** 0.5
    return (q, k, v, out, pages, lengths, c['num_splits'], scale, c['page_size']), {
        'logit_cap': c['logit_cap']}


def expected_outputs(harness, a):
    """Independent float64 oracle evaluated from the plan's pristine clones.

    Tokens are gathered once per request through the logical-page table, each
    query head h reads KV head h // kv_group_num, and every split is a masked
    logsumexp/softmax over its token range. Inactive split slots keep the
    caller-owned bytes of the pristine output clone.
    """
    import torch
    q, k, v, out, pages, lengths = (a[i].detach().to('cpu') for i in range(6))
    splits, scale, page_size = a[6], a[7], a[8]
    cap = a.get('logit_cap', 0.0)
    heads, kv_heads, lv = q.shape[1], k.shape[1], v.shape[-1]
    group = heads // kv_heads
    if group * kv_heads != heads or out.shape[-1] != lv + 1:
        raise AssertionError('Domain control violates the wrapper head/output contract')
    kv_of_head = torch.arange(heads) // group
    q, k, v, result = q.double(), k.double(), v.double(), out.clone()
    for b, length in enumerate(lengths.long().tolist()):
        if length <= 0:
            continue
        positions = torch.arange(length)
        tokens = pages[b].long()[positions // page_size] * page_size + positions % page_size
        keys = k[tokens][:, kv_of_head]      # [tokens, query heads, Lk]
        values = v[tokens][:, kv_of_head]    # [tokens, query heads, Lv]
        logits = torch.einsum('hd,thd->ht', q[b], keys) * scale
        if cap > 0:
            logits = cap * torch.tanh(logits / cap)
        split_of_token = positions // -(-length // splits)
        for split in range(splits):
            selected = split_of_token == split
            if not bool(selected.any()):
                continue
            part = logits[:, selected]
            lse = torch.logsumexp(part, dim=1)
            weights = torch.exp(part - lse[:, None])
            result[b, :, split, :lv] = torch.einsum('ht,thd->hd', weights, values[selected]).float()
            result[b, :, split, lv] = lse.float()
    if not torch.isfinite(result).all():
        raise AssertionError('Domain control oracle is nonfinite')
    return (result.to(a[3].device),)
