"""Hash-pinned CPU validation of the actual MLA paired CSR schedule.

Only data and standard-library code are used. No editable source is imported.
"""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def _equal(actual, expected, message):
    require(canonical(actual) == canonical(expected), message)


def _mla_length(seed, histogram, case_id):
    total = sum(item['weight'] for item in histogram)
    ceiling = (1 << 256) // total * total
    counter = 0
    while True:
        ticket = int.from_bytes(hashlib.sha256(canonical(
            ['mla-observed-kv-distribution-v1', case_id, seed, counter]).encode()).digest(), 'big')
        if ticket < ceiling:
            ticket %= total
            break
        counter += 1
    for item in histogram:
        if ticket < item['weight']:
            return item['length']
        ticket -= item['weight']
    raise ValueError('MLA draw escaped its protected histogram')


def _int32_sha(values):
    require(isinstance(values, list) and all(type(value) is int and -(1 << 31) <= value < (1 << 31)
                                           for value in values), 'Invalid MLA int32 control values')
    return hashlib.sha256(struct.pack('<' + 'i' * len(values), *values)).hexdigest()


def _mla(case, manifest, request, row, entries, registry):
    require(isinstance(registry, dict) and registry.get('schema') == 'mla-observed-control-distribution-v1'
            and case.get('provenance_kind') == 'generated_observed_control_distribution',
            'Verified MLA recipe and distribution case are required')
    require(registry['lengths'] == list(range(192, 201))
            and registry['measurement'] == manifest['measurement']
            and registry['parent_fixture'] == case['fixture']
            and registry['native_source_sha256'] == manifest['native_source_sha256'],
            'MLA recipe scope differs')
    histogram = registry['histogram']
    require([item['length'] for item in histogram] == list(range(192, 201))
            and all(type(item['weight']) is int and item['weight'] > 0 for item in histogram)
            and sum(item['weight'] for item in histogram) == 253952,
            'MLA observed histogram differs')
    parent = row.get('paired_control_registry', {})
    require(set(parent) == {'schema', 'kv_indices', 'kv_indptr'}
            and parent['schema'] == 'mla-parent-control-registry-v1', 'MLA parent control receipt is missing')
    indices, indptr = parent['kv_indices'], parent['kv_indptr']
    require(len(indices) == 16384 and indptr == [200 * i for i in range(65)]
            and all(type(value) is int and 0 <= value < 136856 for value in indices[:12800])
            and indices[12800:] == [-1] * 3584, 'MLA parent control layout differs')
    require(_int32_sha(indices) == registry['parent_controls_sha256']['kv_indices']
            and _int32_sha(indptr) == registry['parent_controls_sha256']['kv_indptr'],
            'MLA parent controls are not bound to the protected recipe')
    geometry = registry['native_inputs']
    require(geometry['kv_indices']['shape'] == [16384] and geometry['kv_indices']['stride'] == [1]
            and geometry['kv_indices']['storage_offset'] == 0 and geometry['kv_indices']['storage_nbytes'] == 65536
            and geometry['kv_indptr']['shape'] == [65] and geometry['kv_indptr']['stride'] == [1]
            and geometry['kv_indptr']['storage_offset'] == 0 and geometry['kv_indptr']['storage_nbytes'] == 260
            and geometry['kv_indices']['dtype'] == geometry['kv_indptr']['dtype'] == 'torch.int32',
            'MLA complete control-buffer ABI differs')
    aliases = {}
    for tensor in geometry.values():
        alias, size = tensor['alias'], tensor['storage_nbytes']
        require(type(size) is int and size > 0 and aliases.get(alias, size) == size,
                'MLA physical storage metadata differs')
        aliases[alias] = size
    private_bytes = sum(aliases.values())
    control_hashes = {}
    for length in registry['lengths']:
        selected = [slot for index in range(64) for slot in indices[200 * index:200 * index + length]]
        selected += [-1] * (16384 - len(selected))
        control_hashes[length] = {'kv_indices_sha256': _int32_sha(selected),
                                  'kv_indptr_sha256': _int32_sha([length * i for i in range(65)])}
    expected = []
    for index in range(len(entries)):
        seed = request['challenge_seed'] + index
        length = _mla_length(seed, histogram, case['case_id'])
        expected.append({'input_seed': seed, 'length': length, **control_hashes[length],
                         'private_cpu_storage_bytes': private_bytes})
    _equal(entries, expected, 'Realized MLA CSR differs from the protected recipe/request draw')
    _equal(row.get('realized_input_signatures'), expected, 'MLA CPU snapshot receipt differs from paired inputs')
    proof = row.get('control_distribution', {})
    policy = manifest['measurement']; warm = policy['warmup_iterations']; count = policy['benchmark_iterations']
    values = [entry['length'] for entry in expected]
    expected_fields = {'kind': 'generated_observed_control_distribution',
        'parent_fixture': registry['parent_fixture'], 'actual_missing_captures_recovered': False,
        'observed_histogram': histogram, 'launch_controls_sha256': fingerprint(registry['source_supported_launches']),
        'selection_trace_sha256': fingerprint(values),
        'runtime_dispatch_mode': 'captured_graph_controls_then_per_replay_tensor_checks',
        'prepared_state_checks': {'candidate': expected, 'reference': expected},
        'warmup_draws': warm, 'measured_draws': count,
        'warmup_histogram': dict(Counter(values[:warm])), 'measured_histogram': dict(Counter(values[warm:])),
        'warmup_values': values[:warm], 'measured_values': values[warm:],
        'private_challenge_sampling': True, 'same_challenge_produces_same_baseline_candidate_states': True}
    _equal({key: proof.get(key) for key in expected_fields}, expected_fields,
           'MLA graph state checks or weighted schedule receipt differs')
    launches = proof.get('runtime_dispatch_calls', [])
    kernels = {item['kernel'] for item in registry['source_supported_launches']}
    require(len(launches) == 4 and len(kernels) == 2
            and {(item.get('leg'), item.get('length'), item.get('kernel')) for item in launches}
                == {(leg, 200, kernel) for leg in ('candidate', 'reference') for kernel in kernels}
            and all(type(item.get('calls')) is int and item['calls'] == 5 for item in launches),
            'MLA calibration and symmetric prepared-graph dispatch evidence differs')


def validate_paired_input_receipts(root, report, manifest, request, provenance):
    root = Path(root).resolve()
    derived = [case for case in manifest['cases'] if 'distribution_recipe' in case]
    require(len(manifest['cases']) == 4 and len(derived) == 1, 'MLA paired task scope differs')
    case = derived[0]; name = case['case_id']
    require(provenance['input_receipt_contracts'][name] == {'kind': 'task_local_paired_inputs_v1'},
            'MLA protected input validator contract differs')
    require(set(provenance['input_signature_fields'][name]) == {
        'input_seed', 'length', 'kv_indices_sha256', 'kv_indptr_sha256', 'private_cpu_storage_bytes'},
        'MLA prepared input signature fields differ')
    reference = case['distribution_recipe']
    require(reference['path'] == 'ut/mla_control_distribution.json', 'MLA recipe path differs')
    path = root / reference['path']
    require(path.is_file() and path.resolve().is_relative_to(root)
            and not any(parent.is_symlink() for parent in (path, *path.parents) if parent == root or root in parent.parents),
            'MLA recipe must be a regular task-contained file')
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == reference['sha256'], 'Manifest-bound MLA recipe digest differs')
    registry = json.loads(data)
    outer = next(row for row in report['cases'] if row['case']['case_id'] == name)
    paired = next(row for row in report['paired_reference_comparison']['cases'] if row['case_id'] == name)
    schedule = paired['input_schedule']
    entries = schedule['warmup_inputs'] + schedule['measured_inputs']
    _mla(case, manifest, request, outer, entries, registry)
    count = manifest['measurement']['benchmark_iterations']
    candidate_samples = outer['samples_ms']
    reference_samples = paired['legs']['protected_reference']['samples_ms']
    require(len(candidate_samples) == len(reference_samples) == count, 'MLA raw paired sample count differs')
    expected_timing = {'schema': 'mla-paired-control-timing-v1',
        'measured_pairs': [{'input_seed': item['input_seed'], 'length': item['length'],
            'candidate_ms': candidate_ms, 'reference_ms': reference_ms}
            for item, candidate_ms, reference_ms in zip(schedule['measured_inputs'], candidate_samples, reference_samples)],
        'candidate_mean_ms': math.fsum(candidate_samples) / count,
        'reference_mean_ms': math.fsum(reference_samples) / count}
    _equal(outer.get('paired_control_timing'), expected_timing,
           'MLA per-state timing receipt differs from all raw paired samples')
    return True
