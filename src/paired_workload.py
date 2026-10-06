"""Data-only reconstruction of declared paired workload receipt protocols.

No candidate or task Python module is imported. Registry bytes are supplied only
after the caller verifies their manifest-bound digest.
"""
from bisect import bisect_right
from collections import Counter
import hashlib
import math
import random
import re
import struct

from .task_contract import canonical, fingerprint, require


def _equal(actual, expected, message):
    require(canonical(actual) == canonical(expected), message)


def _integer_draw(rows, total, seed):
    require(type(total) is int and total > 0, 'Invalid distribution total')
    require(all(type(row['occurrences']) is int and row['occurrences'] > 0 for row in rows)
            and sum(row['occurrences'] for row in rows) == total, 'Invalid distribution frequencies')
    ticket = random.Random(seed).randrange(total)
    running = 0
    for row in rows:
        running += row['occurrences']
        if ticket < running:
            return row
    raise ValueError('Distribution draw escaped its total')


def _moe(case, manifest, request, row, entries, registry):
    require(isinstance(registry, dict) and registry.get('schema') == 'moe-observed-work-distributions-v1',
            'Verified MoE registry is required')
    group = registry['groups'][case['distribution_group']]
    require(group['case_id'] == case['case_id'], 'MoE registry case identity differs')
    histogram = group['histogram']
    require(fingerprint(histogram) == group['histogram_sha256'], 'MoE histogram digest differs')
    for variant in histogram:
        controls = {'num_valid_ids': {'dtype': 'torch.int32', 'shape': [2], 'values': variant['num_valid_ids']}}
        require(fingerprint(controls) == variant['variant_id']
                and variant['num_valid_ids'][1] == group['tokens'], 'MoE observed variant differs')
    expected = []
    for index in range(len(entries)):
        seed = request['challenge_seed'] + index
        selected = _integer_draw(histogram, group['observed_calls'], seed)
        route_seed = int(hashlib.sha256((str(seed) + ':' + selected['variant_id']).encode()).hexdigest()[:16], 16)
        expected.append({'input_seed': seed, 'variant_id': selected['variant_id'],
            'num_valid_ids': selected['num_valid_ids'], 'route_seed': route_seed,
            'routing_origin': 'generated_legal_routes_not_actual_other_rank_capture',
            'activation_origin': 'fresh_seeded_fp8_values_with_remapped_captured_token_scales',
            'fresh_stage1_reference_output': group['seam'] == 'moe2'})
    _equal(entries, expected, 'Realized MoE work differs from the protected registry/request draw')
    policy = manifest['measurement']; warm = policy['warmup_iterations']
    receipt = {'histogram_sha256': group['histogram_sha256'], 'frequency_basis': 'actual eight-rank call counts',
        'sampling': 'integer histogram draws from the protected request challenge seed',
        'schedule_sha256': fingerprint([entry['variant_id'] for entry in expected]),
        'shared_private_request_seed_required_for_paired_comparison': True,
        'warmup_variants': expected[:warm],
        'performance_samples': [dict(entry, device_time_ms=sample) for entry, sample in zip(expected[warm:], row['samples_ms'])],
        'distinct_measured_settings': len({entry['variant_id'] for entry in expected[warm:]}),
        'observed_setting_count': len(histogram), 'all_observed_settings_timed': False,
        'actual_other_rank_routing_recovered': False,
        'scope': 'sampled observed work-count distribution with representative generated routes'}
    _equal(row.get('observed_work_distribution'), receipt,
           'MoE observed-work receipt differs from realized inputs or raw samples')


def _flat(control):
    require(isinstance(control, dict) and set(control) == {'shape', 'runs'}, 'Invalid exact control encoding')
    count = math.prod(control['shape'])
    result = []
    for value, length in control['runs']:
        require(type(value) is int and type(length) is int and length > 0, 'Invalid exact control run')
        result.extend([value] * length)
    require(len(result) == count, 'Control shape/run length differs')
    return result


def _storage_bytes(case):
    sizes = {}
    element_bytes = {'bool': 1, 'uint8': 1, 'int8': 1, 'int16': 2, 'float16': 2, 'bfloat16': 2,
                     'int32': 4, 'float32': 4, 'int64': 8, 'float64': 8}
    for name, tensor in case['tensors'].items():
        if tensor['role'] != 'input':
            continue
        dtype = tensor['dtype'].removeprefix('torch.')
        span = (tensor['storage_offset'] + 1 + sum((n - 1) * s for n, s in zip(tensor['shape'], tensor['strides']))) * element_bytes[dtype]
        size = case.get('original_storage_nbytes', {}).get(name, span)
        require(type(size) is int and size >= span, 'Recorded physical storage is too short')
        alias = case['input_aliases'][name]
        sizes[alias] = max(sizes.get(alias, 0), size)
    return sum(sizes.values())


def _minimax(case, manifest, request, row, entries):
    distribution = case['work_distribution']; variants = sorted(distribution)
    total = 0; cumulative = []
    for variant in variants:
        item = distribution[variant]
        require(set(item) == {'tensor_controls', 'occurrences'}
                and fingerprint(item['tensor_controls']) == variant
                and type(item['occurrences']) is int and item['occurrences'] > 0,
                'Invalid recorded tensor-control distribution')
        for control in item['tensor_controls'].values():
            _flat(control)
        total += item['occurrences']; cumulative.append(total)
    require(total == case['occurrences'], 'Recorded occurrence total differs')
    sampler = 'sha256-domain-seeded-python-randrange-integer-weights-v1'
    distribution_sha = fingerprint(distribution); selected = []
    for index, entry in enumerate(entries):
        seed = request['challenge_seed'] + index
        domain = {'sampler': sampler, 'case_id': case['case_id'],
                  'distribution_sha256': distribution_sha, 'private_seed': seed}
        ticket = random.Random(fingerprint(domain)).randrange(total)
        variant = variants[bisect_right(cumulative, ticket)]; selected.append(variant)
        require(entry['variant_id'] == entry['actual_tensor_controls_sha256'] == variant,
                'Realized tensor controls differ from the protected request draw')
        require(type(entry['private_cpu_storage_bytes']) is int
                and entry['private_cpu_storage_bytes'] == _storage_bytes(case), 'Realized physical storage byte count differs')
        expected_names = {'seq_lens', 'req_to_token', 'slot_ids', 'topk_idx'} & set(case['tensors'])
        addresses = entry['addressing']
        require(set(addresses) == expected_names, 'Realized addressing fields differ')
        for name, address in addresses.items():
            tensor = case['tensors'][name]
            require(set(address) == {'dtype', 'shape', 'sha256'} and address['shape'] == tensor['shape']
                    and address['dtype'].removeprefix('torch.') == tensor['dtype'].removeprefix('torch.')
                    and re.fullmatch('[0-9a-f]{64}', address['sha256']) is not None,
                    'Realized addressing hash or ABI differs')
        lengths = distribution[variant]['tensor_controls'].get('inputs.seq_lens')
        if lengths is not None:
            dtype = case['tensors']['seq_lens']['dtype'].removeprefix('torch.')
            code = {'int32': 'i', 'int64': 'q'}[dtype]
            data = struct.pack('<' + code * math.prod(lengths['shape']), *_flat(lengths))
            require(addresses['seq_lens']['sha256'] == hashlib.sha256(data).hexdigest(),
                    'Realized sequence-length bytes differ from recorded controls')
    # Full addressing hashes are reconstructed by the mandatory pinned CPU
    # validator in the shared loader; this layer independently checks the
    # observed controls, sequence-length bytes, ABI, and storage accounting.
    policy = manifest['measurement']; warm = policy['warmup_iterations']; count = policy['benchmark_iterations']
    measured = selected[warm:]
    expected = {'schema': 'recorded-workload-control-coverage-v2', 'distribution_sha256': distribution_sha,
        'sampler': sampler, 'sampling': 'occurrence_weighted_with_replacement',
        'private_challenge_seed': request['challenge_seed'], 'total_occurrences': total,
        'warmup_variant_ids': selected[:warm], 'measured_variant_ids': measured,
        'measured_histogram': dict(sorted(Counter(measured).items())),
        'schedule_fingerprint': fingerprint({'case_id': case['case_id'], 'seed': request['challenge_seed'],
            'distribution_sha256': distribution_sha, 'selected': selected}),
        'represented_variant_count': len(variants), 'measured_variant_count': len(set(measured)),
        'untimed_variant_ids': sorted(set(variants) - set(measured)),
        'aggregation': 'arithmetic_mean_from_all_100_raw_device_samples',
        'mean_gpu_cost_ms': math.fsum(row['samples_ms']) / count,
        'timing_guarantee': 'sampled_distribution_only', 'exhaustive_timing_claim': False}
    _equal(row.get('workload_control_sampling'), expected,
           'Recorded control receipt differs from request draws or raw samples')


def validate_realized_work(case, manifest, request, row, schedule, contract, *, registry=None):
    entries = schedule['warmup_inputs'] + schedule['measured_inputs']
    kind = contract.get('kind')
    if kind == 'fixed_fixture_v1':
        require(not any(key in case for key in ('distribution_group', 'distribution_recipe', 'work_distribution')),
                'Variable work cannot use the fixed-fixture receipt contract')
        _equal(entries, [{'input_seed': request['challenge_seed'] + i, 'fixture': case['fixture']}
                         for i in range(len(entries))], 'Fixed realized fixture differs from case.fixture')
    elif kind == 'moe_observed_work_v1':
        _moe(case, manifest, request, row, entries, registry)
    elif kind == 'minimax_recorded_controls_v1':
        _minimax(case, manifest, request, row, entries)
    elif kind == 'task_local_paired_inputs_v1':
        # All semantic reconstruction is mandatory in the pinned task-local
        # validator. The caller cannot return a score before that hook passes.
        return
    else:
        raise ValueError('Unsupported realized-work receipt contract')
