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

# Add profiles only after reviewing their protected complete raw-report producer.
# Sharing a report shape is not permission to admit a legacy summary-only task.
_QWEN_GRAPH_PROFILES = {
    'qwen_rmsnorm_graph_v1': {
        'task_type': 'triton2triton',
        'sources': ['source/minimax_m3_rmsnorm.py'],
        'targets': ['gemma_fused_add_rmsnorm', '_gemma_fused_add_rmsnorm_kernel'],
        'op_kind': 'fused_add_rmsnorm', 'cases': 2,
    },
    'qwen_mxfp4_moe_graph_v1': {
        'task_type': 'triton2triton', 'sources': ['source/routing.py'], 'targets': ['select_routes'],
        'op_kind': 'moe', 'cases': 2, 'native_routing': True, 'routing_kind': 'moe',
        'validator': 'scripts/_routing_contract.py',
    },
}


def native_quant_config(config):
    # This is the existing trusted_native_eval contract, not a task-name rule.
    return (config.get('task_type') == 'hip2hip'
            and config.get('source_file_path') == ['source/quant_kernels.cu']
            and config.get('target_kernel_functions') == ['dynamic_per_group_scaled_quant_kernel'])


def quality_required(config):
    return bool(config.get('benchmark_quality') is not None
                or config.get('scoring_baseline') is not None
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
    if config.get('benchmark_quality') is not None:
        return _qwen_graph_cases(root, config, report)
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


def _qwen_graph_cases(root, config, report):
    """Revalidate an explicitly supported Qwen protocol before creating authority.

    A profile does not enable other Qwen report formats. Its protected,
    hash-bound CPU validator supplies the complete existing case/metadata checks;
    candidate code is never imported by this consumer.
    """
    from .native_baseline import read_task_file
    policy = config['benchmark_quality']
    require(isinstance(policy, dict) and set(policy) == {'schema_version', 'adapter', 'validator_sha256'}
            and type(policy['schema_version']) is int and policy['schema_version'] == 1
            and isinstance(policy['adapter'], str) and policy['adapter'] in _QWEN_GRAPH_PROFILES,
            'unsupported benchmark-quality adapter')
    profile = _QWEN_GRAPH_PROFILES[policy['adapter']]
    require(config.get('task_type') == profile['task_type'] and config.get('source_file_path') == profile['sources']
            and config.get('target_kernel_functions') == profile['targets'],
            'Qwen quality adapter target contract differs')
    validator_name = profile.get('validator', 'scripts/task_runner.py')
    validator_path = root / validator_name
    validator = read_task_file(root, validator_name)
    require(hashlib.sha256(validator).hexdigest() == policy['validator_sha256'], 'Qwen quality validator changed')
    metadata = read_task_file(root, 'ut/meta.json')
    manifest = strict_json(metadata.decode())
    require(isinstance(manifest, dict) and manifest.get('op_kind') == profile['op_kind']
            and manifest.get('num_cases') == profile['cases'], 'unsupported Qwen metadata contract')
    sources = {source: hashlib.sha256(read_task_file(root, source)).hexdigest() for source in profile['sources']}
    require(isinstance(report, dict), 'Qwen performance report must be a mapping')
    namespace = {'__file__': str(validator_path.resolve()), '__name__': '_protected_qwen_quality_validator'}
    if profile.get('native_routing'):
        exec(compile(validator, str(validator_path), 'exec'), namespace)
        return _qwen_native_routing_cases(root, profile, report, manifest, metadata, sources, namespace)
    request = report.get('request', {})
    require(report.get('status') == 'ok' and report.get('timer') == 'cuda_graph'
            and report.get('warmup_iterations') == 10 and report.get('benchmark_iterations') == 100,
            'Qwen performance report is incomplete')
    require(isinstance(request, dict) and type(request.get('schema_version')) is int and request['schema_version'] == 1
            and request.get('phase') == 'performance' and isinstance(request.get('request_id'), str)
            and bool(request['request_id']) and request.get('source_sha256') == sources
            and request.get('manifest_sha256') == hashlib.sha256(metadata).hexdigest(),
            'Qwen performance request/source/metadata binding differs')
    manifest_sha256 = request['manifest_sha256']
    case_contract = profile.get('case_contract')
    if case_contract is not None:
        contract_sha256 = hashlib.sha256(read_task_file(root, case_contract)).hexdigest()
        require(request.get('case_contract_sha256') == contract_sha256, 'Qwen protected case contract binding differs')
        manifest_sha256 = fingerprint({'ut/meta.json': request['manifest_sha256'], case_contract: contract_sha256})
    raw = strict_json(read_task_file(root, 'build/_bench_raw.json').decode())
    if case_contract is not None:
        require(isinstance(raw, dict) and raw.get('source_sha256') == sources
                and raw.get('case_contract_sha256') == request['case_contract_sha256'],
                'Qwen raw measurement source/contract binding differs')
    if isinstance(raw, dict) and 'request' in raw:
        require(raw['request'] == request, 'Qwen raw/published invocation differs')
    exec(compile(validator, str(validator_path), 'exec'), namespace)
    validate = namespace[profile.get('validator_function', '_benchmark_cases')]
    rows = validate(raw, root) if profile.get('validator_with_root') else validate(raw)
    require(isinstance(rows, list) and len(rows) == profile['cases'] and report.get('test_cases') == rows,
            'Qwen published cases differ from complete validated raw measurements')
    return _qwen_validated_graph_cases(rows, sources, request['request_id'], manifest_sha256)


def _qwen_native_routing_cases(root, profile, report, manifest, metadata, sources, namespace):
    """Use the protected literal-route validator, never execute selector code."""
    from .native_baseline import read_task_file
    routing = manifest.get('native_routing')
    require(isinstance(routing, dict) and routing.get('kind') == profile['routing_kind'],
            'Qwen native routing kind differs from its registered profile')
    native_path = 'ut/' + manifest['source_provenance']['baseline_ref']
    read_task_file(root, native_path)  # Enforce regular task-contained paths before the validator reads them.
    request = report.get('request')
    require(isinstance(request, dict), 'Qwen native routing request is missing')
    request = namespace['validate_request'](request, root, 'performance')
    source = request['source']
    require(len(sources) == 1 and source['source_file_sha256'] == next(iter(sources.values()))
            and source['meta_sha256'] == hashlib.sha256(metadata).hexdigest(),
            'Qwen native routing source/metadata differs')
    raw = strict_json(read_task_file(root, 'build/reports/' + request['run_id'] + '/raw.json').decode())
    require(isinstance(raw, dict), 'Qwen native routing raw report must be a mapping')
    rows = namespace['validate_payload'](raw, manifest, request)
    rows = [dict(row, test_case_id=row['sig'], execution_time_ms=row['mean_ms']) for row in rows]
    require(len(rows) == profile['cases'] and report.get('test_cases') == rows
            and all(report.get(key) == value for key, value in raw.items()),
            'Qwen native routing publication differs from complete raw measurements')
    # The selector's restricted literal grammar makes this effective identity
    # exact: comments/formatting cannot turn identical GPU routes into a gain.
    return _qwen_validated_graph_cases(rows, source['effective_source_sha256'], request['run_id'],
        source['meta_sha256'], {'benchmark_source_file_sha256': sources,
                               'benchmark_source_identity_kind': 'validated_native_routing'})


def _qwen_validated_graph_cases(rows, sources, request_id, manifest_sha256, extra_metadata=None):
    result = []
    for row in rows:
        require(row.get('benchmark_method') == 'cuda_graph' and 'benchmark_fallback_reason' not in row
                and row.get('benchmark_warmup') == 10 and row.get('benchmark_samples') == 100
                and row.get('benchmark_effective_repeats') == 1
                and isinstance(row.get('validation'), dict)
                and row['validation'].get('measured_graph_validated') is True,
                'Qwen graph method or replay validation is incomplete')
        proof = case_quality(row['test_case_id'], case_ids=[r['test_case_id'] for r in rows],
            sources=sources, request_id=request_id, manifest_sha256=manifest_sha256,
            kind='portable', series={'candidate': row['samples_ms']})
        case = TestCaseResult(row['test_case_id'], execution_time_ms=proof['series']['candidate']['raw_mean_ms'],
            metadata={**{key: value for key, value in row.items() if key not in ('test_case_id', 'execution_time_ms')},
                      **(extra_metadata or {}),
                      'benchmark_quality_required': True, 'benchmark_quality_case': deepcopy(dict(proof))})
        result.append(attach_case_quality(case, proof))
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
