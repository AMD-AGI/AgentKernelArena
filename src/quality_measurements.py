"""Quality adapters for the active protected report contracts; no GPU work."""
from pathlib import Path
from copy import deepcopy
import hashlib

from .benchmark_quality import case_quality, fingerprint, require, validate_case_quality, attach_case_quality
from .task_contract import strict_json, validate_report
from .testcases import TestCaseResult

VARIABLE_FIELDS = {'distribution_group', 'distribution_recipe', 'work_distribution', 'dynamic_work_replay_policy'}
KIMI_FIELDS = {'input_seed', 'work_value', 'actual_control_hashes', 'private_cpu_storage_bytes'}
MLA_FIELDS = {'input_seed', 'length', 'kv_indices_sha256', 'kv_indptr_sha256', 'private_cpu_storage_bytes'}
MINIMAX_FIELDS = {'input_seed', 'variant_id', 'actual_tensor_controls_sha256', 'addressing', 'private_cpu_storage_bytes'}


def native_quant_config(config):
    # This is the existing trusted_native_eval contract, not a task-name rule.
    return (config.get('task_type') == 'hip2hip'
            and config.get('source_file_path') == ['source/quant_kernels.cu']
            and config.get('target_kernel_functions') == ['dynamic_per_group_scaled_quant_kernel'])


def quality_required(config):
    return bool(config.get('scoring_baseline') is not None
                or config.get('trusted_evaluation', {}).get('schema_version') == 1
                or native_quant_config(config))


def require_fixed_work(case):
    require(not (VARIABLE_FIELDS & set(case)), 'variable work requires a validated paired quality adapter')


def validated_work_ids(case, schedule, contract, fields):
    """Project only AFTER all existing registry and pinned CPU hooks pass.

    Projection is fixed by the supported semantic receipt schemas. Addresses,
    seeds and arbitrary reported labels cannot create quality strata.
    """
    kind = contract['kind']
    entries = schedule['measured_inputs']
    if kind == 'fixed_fixture_v1':
        require_fixed_work(case)
        return None
    if (kind in ('moe_observed_work_v1', 'minimax_recorded_controls_v1')
            or (kind == 'task_local_paired_inputs_v1' and set(fields) == MINIMAX_FIELDS and 'work_distribution' in case)):
        require(all(isinstance(x.get('variant_id'), str) and len(x['variant_id']) == 64 for x in entries),
                'validated variant identity is missing')
        if 'work_distribution' in case:
            require(all(x['variant_id'] in case['work_distribution'] for x in entries),
                    'quality work variant is absent from the protected distribution')
        return [fingerprint({'case_id': case['case_id'], 'variant_id': x['variant_id']}) for x in entries]
    require(kind == 'task_local_paired_inputs_v1', 'unknown work-quality receipt kind')
    if set(fields) == KIMI_FIELDS and 'work_distribution' in case:
        field = 'work_value'
    elif set(fields) == MLA_FIELDS and 'distribution_recipe' in case:
        field = 'length'
    else:
        raise ValueError('Benchmark quality: unsupported task-local work-class schema')
    require(all(type(x.get(field)) is int and x[field] >= 0 for x in entries), 'invalid validated work amount')
    return [fingerprint({'case_id': case['case_id'], field: x[field]}) for x in entries]


def portable_test_cases(report, manifest, request, sources):
    measured = validate_report(report, manifest, request)
    require(request.get('phase') == 'performance' and request.get('source_sha256') == sources,
            'portable quality request/source binding differs')
    rows = {row['case']['case_id']: row for row in report['cases']}
    result = []
    for row in measured:
        case = rows[row['test_case_id']]['case']; require_fixed_work(case)
        proof = case_quality(row['test_case_id'], case_ids=[c['case_id'] for c in manifest['cases']],
            sources=sources, request_id=request.get('request_id'),
            manifest_sha256=fingerprint(manifest), kind='portable',
            series={'candidate': rows[row['test_case_id']]['samples_ms']})
        case_result = TestCaseResult(row['test_case_id'], execution_time_ms=row['execution_time_ms'],
            metadata={'params': row['params'], 'case_sha256': row['case_sha256'],
                      'benchmark_method': row['metadata']['benchmark_method'],
                      'benchmark_quality_required': True, 'benchmark_quality_case': deepcopy(dict(proof))})
        result.append(attach_case_quality(case_result, proof))
    return result


def load_quality_measurements(root, config):
    from .native_baseline import read_task_file
    root = Path(root)
    report = strict_json(read_task_file(root, 'build/performance_report.json').decode())
    manifest = strict_json(read_task_file(root, config.get('trusted_evaluation', {}).get('case_manifest', 'cases.json')).decode())
    sources = {name: hashlib.sha256(read_task_file(root, name)).hexdigest() for name in config['source_file_path']}
    if not native_quant_config(config):
        require(config.get('trusted_evaluation', {}).get('schema_version') == 1, 'no protected quality report contract')
        return portable_test_cases(report, manifest, report.get('request', {}), sources)
    from .tools import trusted_native_eval as native
    native.validate_report(report, 'performance', native.identities(root), manifest['cases'])
    result = []
    for row in report['paired_cases']:
        case = next(c for c in manifest['cases'] if c['case_id'] == row['case_id']); require_fixed_work(case)
        candidate = row['timings']['candidate_native']['samples_ms']
        proof = case_quality(row['case_id'], case_ids=[c['case_id'] for c in manifest['cases']],
            sources=sources, request_id=report['run_id'],
            manifest_sha256=fingerprint(manifest), kind='native',
            series={'port': candidate, 'reference': row['timings']['production_native']['samples_ms'], 'candidate': candidate})
        case_result = TestCaseResult(row['case_id'], shape=row['shape'],
            execution_time_ms=row['timings']['candidate_native']['mean_ms'],
            metadata={'params': {'trace_call_count': row['trace_call_count']}, 'benchmark_method': 'cuda_graph',
                      'benchmark_quality_required': True, 'benchmark_quality_case': deepcopy(dict(proof))})
        result.append(attach_case_quality(case_result, proof))
    return result


def workspace_quality(root, config):
    """Authoritative single-run quality for task validation, never a gain claim."""
    from .native_baseline import scoring_policy, load_native_measurements, as_test_cases, read_task_file
    require(quality_required(config), 'task has no supported protected quality contract')
    if scoring_policy(config) is not None:
        cases = as_test_cases(load_native_measurements(root, config))
    else:
        cases = load_quality_measurements(root, config)
    proofs = [validate_case_quality(case) for case in cases]
    require(bool(proofs), 'task validation has no complete timing cases')
    failed = [proof['case_id'] for proof in proofs if proof['status'] != 'pass']
    return {'schema': 'benchmark-quality-v1', 'status': 'reject' if failed else 'pass',
            'comparison_status': 'task_validation_only', 'framework_enforced': True,
            'accepted_gain': False, 'case_count': len(cases), 'failed_cases': failed,
            'benchmark_methods': sorted({(case.metadata or {}).get('benchmark_method') for case in cases}),
            'performance_report_sha256': hashlib.sha256(read_task_file(root, 'build/performance_report.json')).hexdigest(),
            'cases': proofs}
