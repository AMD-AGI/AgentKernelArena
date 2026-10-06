"""Trusted native-production scoring from already measured matched GPU replays."""
from __future__ import annotations

from collections import Counter
import hashlib
import math
from pathlib import Path

from .task_contract import canonical, fingerprint, require, strict_json, validate_report
from .testcases import TestCaseResult


def scoring_policy(config):
    policy = config.get('scoring_baseline')
    if policy is None:
        return None
    require(isinstance(policy, dict) and set(policy) == {'schema_version', 'kind', 'native_source_manifest'}
            and type(policy['schema_version']) is int and policy['schema_version'] == 1
            and policy['kind'] == 'native_production', 'Unsupported scoring baseline policy')
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
    provenance_sha = hashlib.sha256(read_task_file(root, policy['native_source_manifest'])).hexdigest()
    return validate_native_measurements(report, manifest, request, sources, provenance_sha)


def as_test_cases(measured, *, is_baseline=False):
    result = []
    for port, native, candidate in zip(measured['port'], measured['native'], measured['candidate']):
        require(port['case_sha256'] == native['case_sha256'] == candidate['case_sha256'],
                'Native and port case identities differ')
        result.append(TestCaseResult(test_case_id=port['test_case_id'],
            execution_time_ms=native['execution_time_ms'] if is_baseline else candidate['execution_time_ms'],
            metadata={'params': port['params'], 'benchmark_method': native['metadata']['benchmark_method'],
                      'baseline_kind': 'native_production', 'case_sha256': port['case_sha256'],
                      'native_ms': native['execution_time_ms'], 'candidate_ms': candidate['execution_time_ms'],
                      'port_measurement_ms': port['execution_time_ms'],
                      'native_request_id': measured['request_id'],
                      'native_source_hashes': measured['source_sha256'],
                      'native_manifest_sha256': measured['manifest_sha256'],
                      'native_source_manifest_sha256': measured['native_source_manifest_sha256']}))
    return result


def paired_native_cases(cases):
    result = []
    for case in cases:
        metadata = case.metadata or {}
        require(metadata.get('baseline_kind') == 'native_production'
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
        require(b.get('baseline_kind') == c.get('baseline_kind') == 'native_production'
                and b.get('case_sha256') == c.get('case_sha256')
                and b.get('native_manifest_sha256') == c.get('native_manifest_sha256')
                and b.get('native_source_manifest_sha256') == c.get('native_source_manifest_sha256'),
                'Initial and optimized native scoring contracts differ')
        native_ms, candidate_ms = c['native_ms'], c['candidate_ms']
        port_before, port_after = b['port_measurement_ms'], c['port_measurement_ms']
        require(all(type(v) in (int, float) and math.isfinite(v) and v > 0
                    for v in (native_ms, candidate_ms, port_before, port_after)), 'Invalid scoring timing')
        ratio = native_ms / candidate_ms
        rows.append({'test_case_id': case.test_case_id, 'case_sha256': c['case_sha256'],
                     'native_ms': native_ms, 'candidate_ms': candidate_ms, 'speedup': ratio,
                     'port_reference_ms': port_before, 'port_candidate_ms': port_after,
                     'port_to_port_speedup': port_before / port_after,
                     'production_kernel_improvement': ratio > 1, 'regression_vs_native': ratio < 1})
    primary = math.fsum(row['speedup'] for row in rows) / len(rows)
    reference_proof = baseline_cases[0].metadata
    candidate_proof = candidate_cases[0].metadata
    return {'baseline_kind': 'native_production', 'secondary_baseline_kind': 'frozen_port',
            'native_baseline_execution_time': math.fsum(row['native_ms'] for row in rows) / len(rows),
            'native_candidate_execution_time': math.fsum(row['candidate_ms'] for row in rows) / len(rows),
            'native_speedup_ratio': primary,
            'port_to_port_speedup_ratio': math.fsum(row['port_to_port_speedup'] for row in rows) / len(rows),
            'production_kernel_improvement': primary > 1,
            'all_cases_faster_than_native': all(row['production_kernel_improvement'] for row in rows),
            'regressed_case_ids': [row['test_case_id'] for row in rows if row['regression_vs_native']],
            'native_baseline_cases': rows,
            'native_scoring_evidence': {
                'candidate_request_id': candidate_proof['native_request_id'],
                'candidate_source_sha256': candidate_proof['native_source_hashes'],
                'reference_port_request_id': reference_proof['native_request_id'],
                'reference_port_source_sha256': reference_proof['native_source_hashes'],
                'manifest_sha256': candidate_proof['native_manifest_sha256'],
                'native_source_manifest_sha256': candidate_proof['native_source_manifest_sha256'],
            },
            'gain_scope': 'isolated native operator; no end-to-end serving gain asserted'}
