"""Reconstruct Kimi's actual integer work/address controls without GPU imports."""
from array import array
import ast
from collections import Counter
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import random
import sys


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError('Kimi paired inputs: ' + message)


def byte_data(value):
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    require(value.device.type == 'cpu', 'input truth must be CPU-owned')
    return value.numpy().tobytes()


def input_signature(prepared, seed, truth):
    """Hash the real private CPU controls, not an intended generator receipt."""
    metadata = prepared.record['inputs']
    names = (['kv_indptr', 'kv_indices'] if 'kv_indptr' in metadata else
             [name for name in ('num_valid_ids', 'sorted_expert_ids', 'sorted_token_ids', 'topk_ids')
              if metadata.get(name) is not None])
    controls = {}
    buffers = {}
    for name in names:
        meta = metadata[name]
        storage = truth[name if hasattr(prepared, 'reference_inputs') else meta['alias']]
        raw = byte_data(storage)
        offset = meta['storage_offset'] * meta['element_size']
        data = raw[offset:offset + math.prod(meta['shape']) * meta['element_size']]
        buffers[name] = data
        controls[name] = {'dtype': meta['dtype'].removeprefix('torch.'), 'shape': meta['shape'],
                          'sha256': hashlib.sha256(data).hexdigest()}
    require(sys.byteorder == 'little', 'native integer receipt requires little-endian host')
    values = array('i'); values.frombytes(buffers['kv_indptr' if 'kv_indptr' in buffers else 'num_valid_ids'])
    work = values[-1] // 64 if 'kv_indptr' in buffers else values[0]
    size = sum(len(value) if isinstance(value, (bytes, bytearray, memoryview))
               else value.numel() * value.element_size() for value in truth.values())
    return {'input_seed': seed, 'work_value': work, 'actual_control_hashes': controls,
            'private_cpu_storage_bytes': size}


def draw(histogram, rng):
    ticket = rng.randrange(sum(count for _, count in histogram))
    for value, count in histogram:
        if ticket < count:
            return value
        ticket -= count
    raise ValueError('Invalid Kimi work histogram')


@lru_cache(maxsize=4)
def _routes(path, digest):
    """Execute only the three pinned pure routing functions; import no runtime."""
    text = Path(path).read_text()
    require(hashlib.sha256(text.encode()).hexdigest() == digest, 'protected routing source changed')
    names = {'_bounded_counts', 'route_population', 'make_routes'}
    nodes = [node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef) and node.name in names]
    require({node.name for node in nodes} == names and len(nodes) == 3, 'routing function scope changed')
    namespace = {'array': array, 'random': random}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['make_routes']


@lru_cache(maxsize=4)
def _valid_kv(spans):
    rows = array('q')
    for start, count in spans:
        rows.extend(range(start, start + count))
    return rows


def expected_signature(root, case, geometry, seed):
    require(sys.byteorder == 'little', 'native integer receipt requires little-endian host')
    distribution = case['work_distribution']
    field = geometry['histogram_field']
    histogram = distribution[field]
    require(histogram == geometry['histogram'] and sum(n for _, n in histogram) == case['occurrences'],
            'registry histogram differs from all observed case frequencies')
    rng = random.Random(seed); work = draw(histogram, rng)
    if geometry['kind'] == 'moe':
        routes = _routes(str(Path(root)/'ut/routing.py'), geometry['routing_sha256'])(
            geometry['tokens'], geometry['tile_m'], work, geometry['capacity_rows'], geometry['expert_slots'],
            seed, padding_token=geometry['padding_token'])
        values = {name: routes[name] for name in geometry['control_metadata']}
    else:
        valid = _valid_kv(tuple(tuple(span) for span in geometry['valid_kv_spans']))
        require(len(valid) >= work and len(valid) > 0, 'captured KV footprint too small')
        starts = [rng.randrange(len(valid)) for _ in range(64)]
        ids = array('q')
        for start in starts:
            stop = start + work
            ids.extend(valid[start:min(stop, len(valid))])
            if stop > len(valid):
                ids.extend(valid[:stop-len(valid)])
        require(len(ids) <= geometry['kv_capacity'], 'KV index capacity exceeded')
        ids.extend(array('q', [valid[0]]) * (geometry['kv_capacity'] - len(ids)))
        values = {'kv_indices': ids, 'kv_indptr': array('i', (n * work for n in range(65)))}
    controls = {name: {**metadata, 'sha256': hashlib.sha256(values[name].tobytes()).hexdigest()}
                for name, metadata in geometry['control_metadata'].items()}
    return {'input_seed': seed, 'work_value': work, 'actual_control_hashes': controls,
            'private_cpu_storage_bytes': geometry['private_cpu_storage_bytes']}


def sampling_receipt(case, request, signatures, samples, policy):
    field = ('sequence_length_histogram' if 'sequence_length_histogram' in case['work_distribution']
             else 'valid_rows_histogram')
    histogram = case['work_distribution'][field]; warm = policy['warmup_iterations']
    values = [entry['work_value'] for entry in signatures]
    return {'schema': 'kimi-realized-work-sampling-v1', 'histogram_sha256': fingerprint(histogram),
            'request_challenge_seed': request['challenge_seed'], 'sampler': 'python-random-integer-histogram-v1',
            'warmup_values': values[:warm], 'measured_values': values[warm:],
            'measured_histogram': sorted(Counter(values[warm:]).items()),
            'actual_input_schedule_sha256': fingerprint(signatures),
            'mean_gpu_cost_ms': math.fsum(samples) / policy['benchmark_iterations'],
            'observed_setting_count': len(histogram), 'distinct_measured_settings': len(set(values[warm:])),
            'all_observed_settings_timed': False, 'population_frequencies_inferred': False,
            'full_routing_or_address_histories_recovered': False}


def validate_paired_input_receipts(root, report, manifest, request, provenance):
    """Called by the shared consumer after it verifies this helper's source pins."""
    root = Path(root).resolve()
    reference = provenance['work_registry']
    relative = Path(reference['path'])
    path = root/relative
    require(not relative.is_absolute() and '..' not in relative.parts
            and path.resolve().is_relative_to(root) and path.is_file()
            and not any(part.is_symlink() for part in (path, *path.parents) if part != root and root in part.parents),
            'work registry must be a regular task-local file')
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == reference['sha256'], 'work registry digest differs')
    registry = json.loads(data)
    require(registry['schema'] == 'kimi-paired-work-registry-v1', 'work registry schema differs')
    require(set(registry['cases']) == {case['case_id'] for case in manifest['cases']}, 'work registry case scope differs')
    paired = {row['case_id']: row for row in report['paired_reference_comparison']['cases']}
    ordinary = {row['case']['case_id']: row for row in report['cases']}
    policy = manifest['measurement']; total = policy['warmup_iterations'] + policy['benchmark_iterations']
    for case in manifest['cases']:
        name = case['case_id']; geometry = registry['cases'][name]
        require(geometry['case_sha256'] == fingerprint(case) and geometry['fixture'] == case['fixture'],
                'registered geometry is not bound to the exact case and fixture')
        schedule = paired[name]['input_schedule']
        observed = schedule['warmup_inputs'] + schedule['measured_inputs']
        require(len(observed) == total, 'realized schedule count differs')
        expected = []
        for index, actual in enumerate(observed):
            value = expected_signature(root, case, geometry, request['challenge_seed'] + index)
            require(canonical(actual) == canonical(value), 'realized controls differ from registry/request seed')
            expected.append(value)
        require(canonical(ordinary[name].get('realized_input_signatures')) == canonical(expected),
                'private CPU input receipt differs from paired schedule')
        receipt = sampling_receipt(case, request, expected, ordinary[name]['samples_ms'], policy)
        require(canonical(ordinary[name].get('workload_control_sampling')) == canonical(receipt),
                'work receipt differs from actual schedule or raw timing mean')
    return True
