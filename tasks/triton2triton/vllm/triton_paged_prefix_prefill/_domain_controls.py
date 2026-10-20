"""Protected correctness-only controls for the wrapper's supported input domain.

The five scored rows in scripts/task_runner.py use regular lengths, head
dimensions 64/128 and pages of 16/32 tokens. These unscored rows sample the
rest of the domain accepted by the protected `context_attention_fwd` launch:

* float16 data with K-cache vector width x = 8, so head_dim is a multiple of 8;
  head_dim 16..128, where non-power-of-two sizes are padded to the next power
  of two (`BLOCK_DMODEL_PADDED`) and masked by the kernel;
* power-of-two physical pages (BLOCK_M=128, BLOCK_N=64) and other page sizes
  (BLOCK_M=BLOCK_N=32), smaller or larger than the 32-token context tile;
* ragged context/query lengths: empty contexts, empty query segments,
  single-token queries and lengths not aligned with the context tile, the
  query tiles or the page size, including queries spanning two M tiles;
* GQA/MQA head groups, explicit runtime scales and sliding windows.

_arena_checks.py merges this table into CONTRACT_CASES. The adapter binds every
manifest row to it, and each control runs the real public wrapper through the
same protected path as the original controls: the oracle is computed from
pristine clones before the call, reconstructs context from the actual cache
bytes through the page table, and checks every output element with the
original 0.01 gates, followed by the unscored captured-replay diagnostic.
These rows declare only `correctness`; performance never executes them.
"""

KV_VECTOR_WIDTH = 8  # 16-byte float16 vectors, as in setup_paged_kv_cache.
PAGE_MAPPING = ('logical page g, counted row-major over mapped pages, uses physical page '
                '2*(mapped_pages-g); unused table entries use page 1; every page holds '
                'finite random data')

DOMAIN_CASES = {
    'domain_hd80_page8_two_m_tiles': {
        'batch': 3, 'query_heads': 8, 'kv_heads': 2, 'head_dim': 80, 'seed': 1301,
        'dtypes': {'data': 'float16', 'statistics': 'float32', 'routing': 'int32'},
        'contract_case': 'domain_hd80_page8_two_m_tiles',
        'context_lengths': [0, 33, 130], 'query_lengths': [129, 1, 17],
        'sequence_lengths': [129, 34, 147], 'sm_scale': 0.07, 'block_size': 8,
        'kv_vector_width': 8,
        'input_shapes': {'q': [147, 8, 80], 'k': [147, 2, 80], 'v': [147, 2, 80],
                         'k_cache': [45, 2, 10, 8, 8], 'v_cache': [45, 2, 80, 8],
                         'b_loc': [3, 17], 'b_start_loc': [4], 'b_seq_len': [3]},
        'output_shapes': {'o': [147, 8, 80]},
        'page_mapping': PAGE_MAPPING},
    'domain_hd96_page64_mqa_window': {
        'batch': 3, 'query_heads': 4, 'kv_heads': 1, 'head_dim': 96, 'seed': 1302,
        'dtypes': {'data': 'float16', 'statistics': 'float32', 'routing': 'int32'},
        'contract_case': 'domain_hd96_page64_mqa_window',
        'context_lengths': [95, 64, 1], 'query_lengths': [70, 64, 2],
        'sequence_lengths': [165, 128, 3], 'sm_scale': 0.25, 'block_size': 64,
        'sliding_window': 40, 'kv_vector_width': 8,
        'input_shapes': {'q': [136, 4, 96], 'k': [136, 1, 96], 'v': [136, 1, 96],
                         'k_cache': [9, 1, 12, 64, 8], 'v_cache': [9, 1, 96, 64],
                         'b_loc': [3, 2], 'b_start_loc': [4], 'b_seq_len': [3]},
        'output_shapes': {'o': [136, 4, 96]},
        'page_mapping': PAGE_MAPPING},
    'domain_hd112_page12_gqa3_window': {
        'batch': 3, 'query_heads': 6, 'kv_heads': 2, 'head_dim': 112, 'seed': 1303,
        'dtypes': {'data': 'float16', 'statistics': 'float32', 'routing': 'int32'},
        'contract_case': 'domain_hd112_page12_gqa3_window',
        'context_lengths': [47, 0, 25], 'query_lengths': [33, 31, 1],
        'sequence_lengths': [80, 31, 26], 'sm_scale': 0.06, 'block_size': 12,
        'sliding_window': 3, 'kv_vector_width': 8,
        'input_shapes': {'q': [65, 6, 112], 'k': [65, 2, 112], 'v': [65, 2, 112],
                         'k_cache': [15, 2, 14, 12, 8], 'v_cache': [15, 2, 112, 12],
                         'b_loc': [3, 4], 'b_start_loc': [4], 'b_seq_len': [3]},
        'output_shapes': {'o': [65, 6, 112]},
        'page_mapping': PAGE_MAPPING},
    'domain_hd24_page40_wide_window': {
        'batch': 2, 'query_heads': 8, 'kv_heads': 8, 'head_dim': 24, 'seed': 1304,
        'dtypes': {'data': 'float16', 'statistics': 'float32', 'routing': 'int32'},
        'contract_case': 'domain_hd24_page40_wide_window',
        'context_lengths': [300, 81], 'query_lengths': [65, 96],
        'sequence_lengths': [365, 177], 'sm_scale': 0.3, 'block_size': 40,
        'sliding_window': 4096, 'kv_vector_width': 8,
        'input_shapes': {'q': [161, 8, 24], 'k': [161, 8, 24], 'v': [161, 8, 24],
                         'k_cache': [23, 8, 3, 40, 8], 'v_cache': [23, 8, 24, 40],
                         'b_loc': [2, 8], 'b_start_loc': [3], 'b_seq_len': [2]},
        'output_shapes': {'o': [161, 8, 24]},
        'page_mapping': PAGE_MAPPING},
    'domain_hd16_page16_empty_query': {
        'batch': 3, 'query_heads': 2, 'kv_heads': 1, 'head_dim': 16, 'seed': 1305,
        'dtypes': {'data': 'float16', 'statistics': 'float32', 'routing': 'int32'},
        'contract_case': 'domain_hd16_page16_empty_query',
        'context_lengths': [5, 200, 31], 'query_lengths': [0, 3, 40],
        'sequence_lengths': [5, 203, 71], 'sm_scale': 0.5, 'block_size': 16,
        'kv_vector_width': 8,
        'input_shapes': {'q': [43, 2, 16], 'k': [43, 1, 16], 'v': [43, 1, 16],
                         'k_cache': [33, 1, 2, 16, 8], 'v_cache': [33, 1, 16, 16],
                         'b_loc': [3, 13], 'b_start_loc': [4], 'b_seq_len': [3]},
        'output_shapes': {'o': [43, 2, 16]},
        'page_mapping': PAGE_MAPPING},
    'domain_hd128_page32_long_ragged': {
        'batch': 3, 'query_heads': 8, 'kv_heads': 4, 'head_dim': 128, 'seed': 1306,
        'dtypes': {'data': 'float16', 'statistics': 'float32', 'routing': 'int32'},
        'contract_case': 'domain_hd128_page32_long_ragged',
        'context_lengths': [257, 0, 513], 'query_lengths': [255, 7, 1],
        'sequence_lengths': [512, 7, 514], 'sm_scale': 0.05, 'block_size': 32,
        'kv_vector_width': 8,
        'input_shapes': {'q': [263, 8, 128], 'k': [263, 4, 128], 'v': [263, 4, 128],
                         'k_cache': [53, 4, 16, 32, 8], 'v_cache': [53, 4, 128, 32],
                         'b_loc': [3, 17], 'b_start_loc': [4], 'b_seq_len': [3]},
        'output_shapes': {'o': [263, 8, 128]},
        'page_mapping': PAGE_MAPPING},
    'domain_hd32_page128_window': {
        'batch': 2, 'query_heads': 4, 'kv_heads': 4, 'head_dim': 32, 'seed': 1307,
        'dtypes': {'data': 'float16', 'statistics': 'float32', 'routing': 'int32'},
        'contract_case': 'domain_hd32_page128_window',
        'context_lengths': [1, 129], 'query_lengths': [128, 63],
        'sequence_lengths': [129, 192], 'sm_scale': 0.1, 'block_size': 128,
        'sliding_window': 16, 'kv_vector_width': 8,
        'input_shapes': {'q': [191, 4, 32], 'k': [191, 4, 32], 'v': [191, 4, 32],
                         'k_cache': [7, 4, 4, 128, 8], 'v_cache': [7, 4, 32, 128],
                         'b_loc': [2, 2], 'b_start_loc': [3], 'b_seq_len': [2]},
        'output_shapes': {'o': [191, 4, 32]},
        'page_mapping': PAGE_MAPPING},
}


def page_table(case):
    """Return the declared non-identity page table and the physical page count."""
    c = DOMAIN_CASES[case]
    counts = [-(-length // c['block_size']) for length in c['context_lengths']]
    mapped, width = sum(counts), max(counts)
    table, page = [], 0
    for count in counts:
        row = []
        for logical in range(width):
            if logical < count:
                row.append(2 * (mapped - page))
                page += 1
            else:
                row.append(1)
        table.append(row)
    return table, 2 * mapped + 1


def declared_shapes(case):
    """Derive tensor shapes from the semantic fields, independently of input_shapes."""
    c = DOMAIN_CASES[case]
    tokens, batch = sum(c['query_lengths']), c['batch']
    heads, kv_heads, dim = c['query_heads'], c['kv_heads'], c['head_dim']
    x, block = c['kv_vector_width'], c['block_size']
    table, pages = page_table(case)
    return {'q': [tokens, heads, dim], 'k': [tokens, kv_heads, dim], 'v': [tokens, kv_heads, dim],
            'k_cache': [pages, kv_heads, dim // x, block, x], 'v_cache': [pages, kv_heads, dim, block],
            'b_loc': [batch, len(table[0])], 'b_start_loc': [batch + 1], 'b_seq_len': [batch]}


def _validate_table():
    for name, c in DOMAIN_CASES.items():
        lengths = list(zip(c['context_lengths'], c['query_lengths']))
        if (c['contract_case'] != name or len(lengths) != c['batch']
                or c['sequence_lengths'] != [a + b for a, b in lengths]
                or min(min(pair) for pair in lengths) < 0 or max(c['context_lengths']) <= 0
                or max(c['query_lengths']) <= 0 or c['query_heads'] % c['kv_heads']
                or c['kv_vector_width'] != KV_VECTOR_WIDTH or c['head_dim'] % KV_VECTOR_WIDTH
                or not 16 <= c['head_dim'] <= 128 or c['block_size'] <= 0
                or c.get('sliding_window', 1) <= 0):
            raise ValueError(f'Invalid paged-prefix domain control {name}')
        if c['input_shapes'] != declared_shapes(name) or c['output_shapes'] != {'o': c['input_shapes']['q']}:
            raise ValueError(f'Paged-prefix domain control {name} declares inconsistent shapes')
        # An explicit, nondefault runtime scale keeps every comparison on the
        # oracle's independent FP32 cache-gather path (never the generator's
        # FP16 default-scale reference) and rejects kernels ignoring sm_scale.
        if abs(c['sm_scale'] - c['head_dim'] ** -0.5) < 0.02:
            raise ValueError(f'Paged-prefix domain control {name} must use a nondefault scale')


_validate_table()


def control_inputs(case, device='cuda'):
    """Generate one declared control on CPU from its seed, then place it on device."""
    import torch
    c = DOMAIN_CASES[case]
    shapes = c['input_shapes']
    generator = torch.Generator().manual_seed(c['seed'])

    def normal(name):
        values = torch.randn(shapes[name], generator=generator, dtype=torch.float32)
        return values.to(device=device, dtype=torch.float16)

    # Unused pages and page tails hold finite data, so reading past a context
    # length or through an unused table entry changes the attention result.
    q, k, v, k_cache, v_cache = (normal(name) for name in ('q', 'k', 'v', 'k_cache', 'v_cache'))
    table, _ = page_table(case)
    starts = [0]
    for length in c['query_lengths']:
        starts.append(starts[-1] + length)
    routing = {'device': device, 'dtype': torch.int32}
    args = (q, k, v, torch.zeros_like(q), k_cache, v_cache, torch.tensor(table, **routing),
            torch.tensor(starts, **routing), torch.tensor(c['sequence_lengths'], **routing))
    names = ('q', 'k', 'v', None, 'k_cache', 'v_cache', 'b_loc', 'b_start_loc', 'b_seq_len')
    if any(name is not None and list(value.shape) != shapes[name] for name, value in zip(names, args)):
        raise AssertionError(f'Paged-prefix domain control {case} differs from its declaration')
    kwargs = {'max_input_len': max(c['query_lengths']), 'sm_scale': c['sm_scale']}
    if 'sliding_window' in c:
        kwargs['sliding_window'] = c['sliding_window']
    return args, kwargs
