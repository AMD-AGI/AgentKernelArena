"""Trusted native-production scoring from already measured matched GPU replays."""
from __future__ import annotations

from collections import Counter
import hashlib
import math
from pathlib import Path
from threading import RLock

from .task_contract import canonical, fingerprint, require, strict_json, validate_report
from .testcases import TestCaseResult

BASELINE_KINDS = {'native_production', 'protected_reference'}
_INPUT_VALIDATOR_LOCK = RLock()

def scoring_policy(config):
    policy = config.get('scoring_baseline')
    if policy is None:
        return None
    require(isinstance(policy, dict) and set(policy) == {'schema_version', 'kind', 'native_source_manifest'}
            and type(policy['schema_version']) is int and policy['schema_version'] in (1, 2)
            and policy['kind'] in BASELINE_KINDS
            and (policy['schema_version'] == 2 or policy['kind'] == 'native_production'),
            'Unsupported scoring baseline policy')
    require(config.get('trusted_evaluation', {}).get('schema_version') == 1,
            'Native baseline scoring requires a trusted evaluation contract')
    relative = Path(policy['native_source_manifest'])
    require(not relative.is_absolute() and '..' not in relative.parts and len(relative.parts) > 1
            and relative.parts[0] == 'provenance', 'Native source manifest must be protected task provenance')
    return policy


def read_task_file(root, relative):
    root = Path(root).resolve()
    relative = Path(relative)
    path = root / relative
    require(not relative.is_absolute() and '..' not in relative.parts
            and path.resolve().is_relative_to(root), 'Native scoring input escapes task')
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents) if p != root and root in p.parents),
            'Native scoring input must be a regular task-contained file')
    return path.read_bytes()


def _validate_task_input_receipts(root, report, manifest, request, provenance, source_hashes):
    """Run only the hash-pinned protected CPU validator and its declared helpers."""
    with _INPUT_VALIDATOR_LOCK:
        return _run_task_input_validator(root, report, manifest, request, provenance, source_hashes)


def _run_task_input_validator(root, report, manifest, request, provenance, source_hashes):
    import importlib.abc
    import importlib.machinery
    import importlib.util
    import sys
    root = Path(root).resolve()
    hook = provenance.get('paired_input_validator', {})
    require(hook.get('schema') == 'task-local-paired-input-validator-v1',
            'A protected task input validator is required')
    files = hook.get('files_sha256', {})
    entry = hook.get('validator')
    require(isinstance(files, dict) and entry in files and files and not (set(files) & set(source_hashes)),
            'Input validator closure is missing or contains candidate sources')
    verified_sources = {}
    module_paths = {}
    for name, expected in files.items():
        require(Path(name).parts[0] == 'ut' and name.endswith('.py'), 'Input validator must be protected task Python')
        source = read_task_file(root, name)
        require(hashlib.sha256(source).hexdigest() == expected,
                'Protected input validator dependency changed: ' + name)
        verified_sources[name] = source
        module_name = '.'.join(Path(name).relative_to('ut').with_suffix('').parts)
        module_paths[module_name.removesuffix('.__init__')] = name
    class VerifiedSourceLoader(importlib.abc.Loader):
        def __init__(self, relative):
            self.relative = relative
        def create_module(self, spec):
            return None
        def get_filename(self, fullname):
            return str(root / self.relative)
        def is_package(self, fullname):
            return Path(self.relative).name == '__init__.py'
        def get_source(self, fullname):
            return importlib.util.decode_source(verified_sources[self.relative])
        def exec_module(self, module):
            # Never ask SourceFileLoader for code: it can read a matching .pyc
            # even while dont_write_bytecode is true. Execute the exact bytes
            # whose digest was checked above, for entries and dependencies.
            code = compile(verified_sources[self.relative], str(root / self.relative),
                           'exec', dont_inherit=True)
            exec(code, module.__dict__)
    def source_spec(fullname, relative):
        path = root / relative
        locations = [str(path.parent)] if path.name == '__init__.py' else None
        return importlib.util.spec_from_file_location(fullname, path,
            loader=VerifiedSourceLoader(relative), submodule_search_locations=locations)
    class Guard(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname in module_paths:
                return source_spec(fullname, module_paths[fullname])
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is not None and spec.origin not in (None, 'built-in', 'frozen'):
                origin = Path(spec.origin).resolve()
                if origin.is_relative_to(root):
                    relative = origin.relative_to(root).as_posix()
                    require(relative in verified_sources,
                            'Input validator attempted an undeclared or candidate task import')
                    return source_spec(fullname, relative)
            return spec
    stems = set(module_paths)
    # Remove cached task packages and helper basenames as well as full names;
    # an ambient module must not win before our verified finder is consulted.
    for name in module_paths:
        pieces = name.split('.')
        stems.update('.'.join(pieces[:index]) for index in range(1, len(pieces) + 1))
        stems.add(pieces[-1])
    def belongs_to_task(module):
        file = getattr(module, '__file__', None)
        return isinstance(file, str) and Path(file).resolve().is_relative_to(root)
    # A previously imported candidate module must not bypass the import guard.
    saved = {name: module for name, module in list(sys.modules.items())
             if name in stems or belongs_to_task(module)}
    for name in saved:
        del sys.modules[name]
    alias = '_aka_paired_input_validator_' + fingerprint(hook)
    guard = Guard(); old_path = list(sys.path); old_bytecode = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(root / 'ut')); sys.meta_path.insert(0, guard)
        spec = source_spec(alias, entry)
        module = importlib.util.module_from_spec(spec); sys.modules[alias] = module
        spec.loader.exec_module(module)
        function = getattr(module, hook.get('function', ''), None)
        require(callable(function), 'Protected input validator function is missing')
        result = function(root, report, manifest, request, provenance)
        require(result is None or result is True or (isinstance(result, dict) and result.get('status') == 'ok'),
                'Protected input validator rejected the receipt')
        for name, expected in files.items():
            require(hashlib.sha256(read_task_file(root, name)).hexdigest() == expected,
                    'Input validator dependency changed during validation')
    finally:
        sys.path[:] = old_path
        sys.dont_write_bytecode = old_bytecode
        if guard in sys.meta_path:
            sys.meta_path.remove(guard)
        sys.modules.pop(alias, None)
        for name, module in list(sys.modules.items()):
            if name in stems or belongs_to_task(module):
                sys.modules.pop(name, None)
        sys.modules.update(saved)


def validate_native_measurements(report, manifest, request, source_hashes, native_source_manifest_sha256):
    """Derive all means/ratios from full raw rows; never trust supplied speedups."""
    require(request.get('phase') == 'performance' and isinstance(request.get('request_id'), str)
            and bool(request['request_id']) and request.get('source_sha256') == source_hashes,
            'Native comparison does not bind the current performance source/request')
    port = validate_report(report, manifest, request)
    native = report.get('native_production_comparison')
    require(isinstance(native, dict) and native.get('schema') == 'native-production-comparison-v1'
            and type(native.get('schema_version')) is int and native['schema_version'] == 1
            and native.get('status') == 'ok'
            and native.get('baseline_kind') == 'native_production'
            and native.get('score_input') is True and native.get('diagnostic_only') is False,
            'Missing scoreable native-production comparison')
    require(canonical(native.get('request')) == canonical(request)
            and native.get('source_hashes') == source_hashes
            and native.get('source_sha256') in source_hashes.values()
            and native.get('manifest_sha256') == fingerprint(manifest)
            and native.get('runtime_image') == manifest['runtime_image']
            and native.get('native_source_manifest_sha256') == native_source_manifest_sha256,
            'Stale native comparison source, request, cases, image or provenance')
    require(type(native.get('challenge_seed')) is int and native['challenge_seed'] >= 0,
            'Native comparison must record its actual private challenge seed')
    rows = native.get('cases')
    require(isinstance(rows, list) and all(isinstance(row, dict) for row in rows),
            'Malformed native comparison cases')
    require(Counter(row.get('case_id') for row in rows) == Counter(case['case_id'] for case in manifest['cases']),
            'Native comparison case coverage differs')
    for row in rows:
        require(row.get('identical_captured_ABI_and_fresh_numeric_challenge_sequence') is True
                and (row.get('native_output_parity') is True or row.get('native_output_conformance') is True),
                'Native comparison lacks matched-input numerical conformance')
        require(set(row.get('legs', {})) == {'native_production', 'candidate_port'},
                'Native comparison legs differ')
        require(all(leg.get('case', {}).get('case_id') == row['case_id'] for leg in row['legs'].values()),
                'Native leg identity differs from its case')
    measured = {}
    for leg in ('native_production', 'candidate_port'):
        measured[leg] = validate_report({'schema_version': 1, 'status': 'ok', 'request': request,
                                        'cases': [row['legs'][leg] for row in rows]}, manifest, request)
    return {'port': port, 'native': measured['native_production'], 'candidate': measured['candidate_port'],
            'request_id': request['request_id'], 'source_sha256': source_hashes,
            'comparison_challenge_seed': native['challenge_seed'],
            'manifest_sha256': fingerprint(manifest), 'native_source_manifest_sha256': native_source_manifest_sha256}


def load_native_measurements(root, config, *, report=None, request=None):
    policy = scoring_policy(config)
    require(policy is not None, 'Task did not opt into native baseline scoring')
    root = Path(root)
    manifest = strict_json(read_task_file(root, config['trusted_evaluation'].get('case_manifest', 'cases.json')).decode())
    if report is None:
        report = strict_json(read_task_file(root, 'build/performance_report.json').decode())
    request = report.get('request', {}) if request is None else request
    sources = {name: hashlib.sha256(read_task_file(root, name)).hexdigest() for name in config['source_file_path']}
    provenance_bytes = read_task_file(root, policy['native_source_manifest'])
    provenance_sha = hashlib.sha256(provenance_bytes).hexdigest()
    if policy['schema_version'] == 2:
        provenance = strict_json(provenance_bytes.decode())
        references = config['trusted_evaluation']['reference_sources']
        expected = {name: hashlib.sha256(read_task_file(root, path)).hexdigest()
                    for name, path in references.items()}
        require(provenance['reference_source_sha256'] == expected,
                'Protected reference source provenance changed')
        return validate_paired_measurements(report, manifest, request, sources, provenance_sha, policy, provenance,
                                            task_root=root)
    return validate_native_measurements(report, manifest, request, sources, provenance_sha)


def validate_paired_measurements(report, manifest, request, source_hashes, provenance_sha, policy, provenance,
                                 *, task_root=None):
    """Validate reuse of the timed oracle and actual input pairing for every case."""
    port = validate_report(report, manifest, request)
    require(request.get('phase') == 'performance' and request.get('source_sha256') == source_hashes,
            'Paired comparison request/source differs')
    paired = report.get('paired_reference_comparison', {})
    require(paired.get('schema') == 'paired-reference-comparison-v1'
            and type(paired.get('schema_version')) is int and paired['schema_version'] == 1
            and paired.get('status') == 'ok' and paired.get('score_input') is True
            and paired.get('diagnostic_only') is False, 'Missing scoreable paired reference comparison')
    require(paired.get('baseline_kind') == provenance.get('baseline_kind') == policy['kind']
            and canonical(paired.get('request')) == canonical(request)
            and paired.get('source_hashes') == source_hashes
            and paired.get('manifest_sha256') == fingerprint(manifest)
            and paired.get('runtime_image') == manifest['runtime_image']
            and paired.get('native_source_manifest_sha256') == provenance_sha,
            'Paired reference identity or provenance differs')
    seed = request.get('challenge_seed')
    require(type(seed) is int and seed >= 0 and paired.get('challenge_seed') == seed,
            'Paired comparison challenge differs')
    require(paired.get('anti_cheat_attestation') is False and paired.get('fresh_trusted_host_retest_required') is True,
            'Ordinary paired timing cannot replace the trusted host retest')
    rows = paired.get('cases', [])
    require(isinstance(rows, list) and all(isinstance(row, dict) for row in rows), 'Malformed paired rows')
    require(Counter(row.get('case_id') for row in rows) == Counter(case['case_id'] for case in manifest['cases']),
            'Paired reference case coverage differs')
    ordinary = {row['case']['case_id']: row for row in report['cases']}
    signature_fields = provenance.get('input_signature_fields', {})
    require(set(signature_fields) == set(ordinary), 'Realized input signature scope differs')
    receipt_contracts = provenance.get('input_receipt_contracts', {})
    require(set(receipt_contracts) == set(ordinary), 'Realized-work receipt contract scope differs')
    registry = None
    if any(contract.get('kind') == 'moe_observed_work_v1' for contract in receipt_contracts.values()):
        require(task_root is not None, 'A task root is required to verify the manifest-bound work registry')
        reference = manifest.get('observed_work_distributions', {})
        registry_bytes = read_task_file(task_root, reference.get('path', ''))
        require(hashlib.sha256(registry_bytes).hexdigest() == reference.get('sha256'),
                'Manifest-bound work registry digest differs')
        registry = strict_json(registry_bytes.decode())
    for fields in signature_fields.values():
        require(isinstance(fields, list) and len(fields) >= 2 and len(set(fields)) == len(fields)
                and 'input_seed' in fields and all(isinstance(field, str) and field for field in fields),
                'Realized input signature fields are missing')
    compiled_rows = report.get('compiled_specializations', [])
    require(Counter(row.get('case_id') for row in compiled_rows) == Counter(ordinary.keys()),
            'Compiled paired reference coverage differs')
    compiled = {row['case_id']: row for row in compiled_rows}
    p = manifest['measurement']; count = p['benchmark_iterations']; warmups = p['warmup_iterations']
    schedules = {}
    core = ('case', 'correct', 'samples_ms', 'warmup_iterations', 'fresh_input_resets',
            'output_initializations', 'oracle_checks', 'benchmark_method')
    for row in rows:
        name = row['case_id']; outer = ordinary[name]
        require(row.get('candidate_snapshot_before_reference') is True
                and row.get('reference_reused_as_oracle') is True and row.get('output_conformance') is True
                and row.get('checked_pair_count') == warmups + count
                and row.get('reference_output_clear_calls') == warmups + count,
                'Paired oracle ordering, counts or output clearing differ')
        require(row.get('graph_setup') == {'warmup_invocations_per_leg': 3, 'capture_invocations_per_leg': 1,
                'capture_on_warmed_stream': True, 'separate_graphs_and_outputs': True},
                'Paired graph setup differs')
        require(set(row.get('legs', {})) == {'candidate_port', 'protected_reference'}, 'Paired legs differ')
        candidate = row['legs']['candidate_port']; reference = row['legs']['protected_reference']
        require(all(candidate.get(field) == outer.get(field) for field in core),
                'Candidate samples were replaced or measured twice')
        schedule = row.get('input_schedule', {})
        require(schedule.get('schema_version') == 1 and schedule.get('case_id') == name
                and schedule.get('manifest_sha256') == fingerprint(manifest), 'Input schedule scope differs')
        warm = schedule.get('warmup_inputs', []); measured = schedule.get('measured_inputs', [])
        require(len(warm) == warmups and len(measured) == count,
                'Input schedule draw counts differ')
        require(all(isinstance(item, dict) and set(item) == set(signature_fields[name])
                    and type(item.get('input_seed')) is int and item['input_seed'] == seed + index
                    for index, item in enumerate(warm + measured)), 'Input schedule is not the request sequence')
        schedules[name] = fingerprint(schedule)
        require(candidate.get('paired_schedule_sha256') == reference.get('paired_schedule_sha256')
                == outer.get('paired_schedule_sha256') == schedules[name], 'Paired input schedules differ')
        from .paired_workload import validate_realized_work
        validate_realized_work(outer['case'], manifest, request, outer, schedule, receipt_contracts[name],
                               registry=registry)
        cb, rb = row.get('candidate_binding', {}), row.get('reference_binding', {})
        require(cb == compiled[name].get('candidate_binding') and rb == compiled[name].get('reference_binding')
                and compiled[name].get('invoked_and_synchronized') is True,
                'Paired specialization bindings differ')
        require(cb.get('source_sha256') == source_hashes
                and rb.get('source_sha256') == provenance['reference_source_sha256']
                and rb.get('leg') == 'reference' and cb.get('module') != rb.get('module')
                and cb.get('gpu_binding') == rb.get('gpu_binding') == provenance['gpu_binding'],
                'Protected reference or candidate source binding differs')
    legs = {label: validate_report({'schema_version': 1, 'status': 'ok', 'request': request,
            'cases': [row['legs'][label] for row in rows]}, manifest, request)
            for label in ('candidate_port', 'protected_reference')}
    if any(contract.get('kind') in ('minimax_recorded_controls_v1', 'task_local_paired_inputs_v1')
           for contract in receipt_contracts.values()):
        require(task_root is not None, 'A task root is required for full addressing reconstruction')
        _validate_task_input_receipts(task_root, report, manifest, request, provenance, source_hashes)
    return {'port': port, 'native': legs['protected_reference'], 'candidate': legs['candidate_port'],
            'request_id': request['request_id'], 'source_sha256': source_hashes,
            'comparison_challenge_seed': seed, 'manifest_sha256': fingerprint(manifest),
            'native_source_manifest_sha256': provenance_sha, 'baseline_kind': policy['kind'],
            'paired_input_schedules': schedules, 'comparison_protocol_version': 2}


def as_test_cases(measured, *, is_baseline=False):
    result = []
    for port, native, candidate in zip(measured['port'], measured['native'], measured['candidate']):
        require(port['case_sha256'] == native['case_sha256'] == candidate['case_sha256'],
                'Native and port case identities differ')
        result.append(TestCaseResult(test_case_id=port['test_case_id'],
            execution_time_ms=native['execution_time_ms'] if is_baseline else candidate['execution_time_ms'],
            metadata={'params': port['params'], 'benchmark_method': native['metadata']['benchmark_method'],
                      'baseline_kind': measured.get('baseline_kind', 'native_production'), 'case_sha256': port['case_sha256'],
                      'native_ms': native['execution_time_ms'], 'candidate_ms': candidate['execution_time_ms'],
                      'port_measurement_ms': port['execution_time_ms'],
                      'native_request_id': measured['request_id'],
                      'native_comparison_challenge_seed': measured['comparison_challenge_seed'],
                      'native_source_hashes': measured['source_sha256'],
                      'native_manifest_sha256': measured['manifest_sha256'],
                      'native_source_manifest_sha256': measured['native_source_manifest_sha256'],
                      'paired_input_schedule_sha256': measured.get('paired_input_schedules', {}).get(port['test_case_id']),
                      'comparison_protocol_version': measured.get('comparison_protocol_version', 1)}))
    return result


def paired_native_cases(cases):
    result = []
    for case in cases:
        metadata = case.metadata or {}
        require(metadata.get('baseline_kind') in BASELINE_KINDS
                and isinstance(metadata.get('native_request_id'), str) and bool(metadata['native_request_id']),
                'Native scoring cases lack validated comparison evidence')
        timing = metadata.get('native_ms')
        require(type(timing) in (int, float) and math.isfinite(timing) and timing > 0,
                'Invalid native baseline timing')
        result.append(TestCaseResult(case.test_case_id, case.shape, timing, dict(metadata)))
    return result


def metric_summary(baseline_cases, candidate_cases):
    """Keep the old frozen-port ratio secondary to matched production speedup."""
    before = {case.test_case_id: case for case in baseline_cases}
    require(bool(before) and len(before) == len(baseline_cases) == len(candidate_cases)
            and set(before) == {case.test_case_id for case in candidate_cases},
            'Native scoring requires complete initial and optimized case sets')
    for series in (baseline_cases, candidate_cases):
        require(len({(case.metadata or {}).get('native_request_id') for case in series}) == 1,
                'Native scoring mixed cases from different requests')
    rows = []
    for case in candidate_cases:
        b, c = before[case.test_case_id].metadata or {}, case.metadata or {}
        require(b.get('baseline_kind') == c.get('baseline_kind') and c.get('baseline_kind') in BASELINE_KINDS
                and b.get('case_sha256') == c.get('case_sha256')
                and b.get('native_manifest_sha256') == c.get('native_manifest_sha256')
                and b.get('native_source_manifest_sha256') == c.get('native_source_manifest_sha256'),
                'Initial and optimized native scoring contracts differ')
        native_ms, candidate_ms = c['native_ms'], c['candidate_ms']
        port_before, port_after = b['port_measurement_ms'], c['port_measurement_ms']
        require(all(type(v) in (int, float) and math.isfinite(v) and v > 0
                    for v in (native_ms, candidate_ms, port_before, port_after)), 'Invalid scoring timing')
        ratio = native_ms / candidate_ms
        secondary_paired = (type(b.get('native_comparison_challenge_seed')) is int
                            and b['native_comparison_challenge_seed'] == c.get('native_comparison_challenge_seed'))
        if b.get('comparison_protocol_version', 1) == 2 or c.get('comparison_protocol_version', 1) == 2:
            secondary_paired = secondary_paired and bool(b.get('paired_input_schedule_sha256')) and (
                b['paired_input_schedule_sha256'] == c.get('paired_input_schedule_sha256'))
        rows.append({'test_case_id': case.test_case_id, 'case_sha256': c['case_sha256'],
                     'native_ms': native_ms, 'candidate_ms': candidate_ms, 'speedup': ratio,
                     'port_reference_ms': port_before, 'port_candidate_ms': port_after,
                     'port_to_port_speedup': port_before / port_after if secondary_paired else None,
                     'secondary_comparison_status': 'paired' if secondary_paired else 'unpaired_workload',
                     'production_kernel_improvement': c['baseline_kind'] == 'native_production' and ratio > 1,
                     'protected_reference_improvement': ratio > 1,
                     'regression_vs_native': c['baseline_kind'] == 'native_production' and ratio < 1,
                     'regression_vs_reference': ratio < 1})
    primary = math.fsum(row['speedup'] for row in rows) / len(rows)
    reference_proof = baseline_cases[0].metadata
    candidate_proof = candidate_cases[0].metadata
    kind = candidate_proof['baseline_kind']
    all_secondary_paired = all(row['port_to_port_speedup'] is not None for row in rows)
    return {'baseline_kind': kind, 'secondary_baseline_kind': 'frozen_port',
            'native_baseline_execution_time': math.fsum(row['native_ms'] for row in rows) / len(rows),
            'native_candidate_execution_time': math.fsum(row['candidate_ms'] for row in rows) / len(rows),
            'native_speedup_ratio': primary,
            'port_to_port_speedup_ratio': (math.fsum(row['port_to_port_speedup'] for row in rows) / len(rows)
                                          if all_secondary_paired else None),
            'secondary_comparison_status': 'paired' if all_secondary_paired else 'unpaired_workload',
            'production_kernel_improvement': kind == 'native_production' and primary > 1,
            'protected_reference_improvement': primary > 1,
            'all_cases_faster_than_native': kind == 'native_production' and all(row['production_kernel_improvement'] for row in rows),
            'regressed_case_ids': [row['test_case_id'] for row in rows if row['regression_vs_reference']],
            'native_baseline_cases': rows,
            'native_scoring_evidence': {
                'candidate_request_id': candidate_proof['native_request_id'],
                'candidate_comparison_challenge_seed': candidate_proof['native_comparison_challenge_seed'],
                'candidate_source_sha256': candidate_proof['native_source_hashes'],
                'reference_port_request_id': reference_proof['native_request_id'],
                'reference_port_comparison_challenge_seed': reference_proof['native_comparison_challenge_seed'],
                'reference_port_source_sha256': reference_proof['native_source_hashes'],
                'manifest_sha256': candidate_proof['native_manifest_sha256'],
                'native_source_manifest_sha256': candidate_proof['native_source_manifest_sha256'],
            },
            'gain_scope': ('isolated native-production operator; no end-to-end serving gain asserted'
                           if kind == 'native_production' else
                           'isolated operator versus the declared protected reference; no native-production or end-to-end gain asserted')}
