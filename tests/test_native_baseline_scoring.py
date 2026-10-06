"""Production gain cannot be inferred from improvement to a slower supplied port."""
import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from src import evaluator, performance
from src.native_baseline import (
    as_test_cases, load_native_measurements, metric_summary, scoring_policy,
    validate_native_measurements,
)
from src.score import score
from src.task_contract import finalize_report, fingerprint


POLICY = {'schema_version': 1, 'kind': 'native_production',
          'native_source_manifest': 'provenance/NATIVE.json'}


def example(*, source_hash='a' * 64, request_id='fresh', port_ms=4.0, native_ms=1.0, candidate_ms=4.0):
    tensor = {'role': 'input', 'shape': [4], 'strides': [1], 'storage_offset': 0,
              'dtype': 'bfloat16', 'device_type': 'cuda'}
    manifest = {'schema_version': 1, 'runtime_image': 'image@sha256:' + 'b' * 64,
                'cases': [{'case_id': name, 'calls_per_sample': 1, 'occurrences': count,
                           'tensors': {'A': tensor, 'C': {**tensor, 'role': 'output'}},
                           'scalars': {'K': 4}} for name, count in [('decode', 3), ('prefill', 9)]],
                'measurement': {'method': 'cuda_graph', 'warmup_iterations': 1, 'benchmark_iterations': 2,
                                'correctness_seeds': [42, 43], 'negative_controls': ['no_op', 'wrong_output'],
                                'refresh_inputs': 'each_replay', 'initialize_outputs': 'each_replay',
                                'validate_outputs': 'each_replay'}}
    request = {'schema_version': 1, 'request_id': request_id, 'phase': 'performance',
               'source_sha256': {'source/kernel.py': source_hash}, 'manifest_sha256': fingerprint(manifest),
               'package_sha256': 'c' * 64, 'challenge_seed': 42}
    def row(case, timing):
        return {'case': copy.deepcopy(case), 'correct': True, 'samples_ms': [timing, timing],
                'warmup_iterations': 1, 'fresh_input_resets': 2, 'output_initializations': 2,
                'oracle_checks': 2, 'benchmark_method': 'cuda_graph'}
    report = finalize_report({'schema_version': 1, 'status': 'ok', 'request': request,
                              'cases': [row(case, port_ms) for case in manifest['cases']]}, manifest, request)
    report['native_production_comparison'] = {
        'schema_version': 1, 'schema': 'native-production-comparison-v1', 'status': 'ok',
        'baseline_kind': 'native_production', 'score_input': True, 'diagnostic_only': False,
        'request': copy.deepcopy(request), 'source_hashes': request['source_sha256'], 'source_sha256': source_hash,
        'manifest_sha256': fingerprint(manifest), 'runtime_image': manifest['runtime_image'],
        'native_source_manifest_sha256': 'd' * 64, 'challenge_seed': 731,
        'cases': [{'case_id': case['case_id'], 'native_output_parity': True,
                   'identical_captured_ABI_and_fresh_numeric_challenge_sequence': True,
                   'speedup_vs_native': 999999, 'candidate_faster_than_native': True,
                   'legs': {'candidate_port': row(case, candidate_ms), 'native_production': row(case, native_ms)}}
                  for case in manifest['cases']]}
    return report, manifest, request


def measured(*args, **kwargs):
    report, manifest, request = example(*args, **kwargs)
    return validate_native_measurements(report, manifest, request, request['source_sha256'], 'd' * 64)


def test_twice_faster_port_still_slower_than_production_does_not_earn_a_production_gain():
    before = as_test_cases(measured(port_ms=4, candidate_ms=4), is_baseline=True)
    after = as_test_cases(measured(source_hash='e' * 64, request_id='optimized', port_ms=2, candidate_ms=2))
    summary = metric_summary(before, after)
    assert summary['port_to_port_speedup_ratio'] == 2
    assert summary['native_speedup_ratio'] == 0.5
    assert summary['production_kernel_improvement'] is False
    assert summary['all_cases_faster_than_native'] is False
    assert summary['regressed_case_ids'] == ['decode', 'prefill']
    assert summary['native_scoring_evidence']['candidate_comparison_challenge_seed'] == 731
    assert summary['native_scoring_evidence']['reference_port_comparison_challenge_seed'] == 731
    assert score(True, True, 1, 2, speedup_ratio=summary['native_speedup_ratio'],
                 benchmark_method_consistent=True) == 170


def test_aggregate_gain_and_individual_regressions_are_both_explicit():
    before = as_test_cases(measured(), is_baseline=True)
    after = as_test_cases(measured(port_ms=2, candidate_ms=2))
    after[0].metadata['native_ms'] = 4
    summary = metric_summary(before, after)
    assert summary['native_speedup_ratio'] == 1.25
    assert summary['production_kernel_improvement'] is True
    assert summary['all_cases_faster_than_native'] is False
    assert summary['regressed_case_ids'] == ['prefill']


@pytest.mark.parametrize('attack', ['missing', 'old_request', 'wrong_source', 'wrong_image', 'wrong_provenance',
                                   'missing_case', 'duplicate_case', 'changed_shape', 'changed_count',
                                   'missing_sample', 'wrong_method', 'missing_check', 'unmatched_inputs',
                                   'unqualified_native', 'diagnostic_only'])
def test_native_evidence_fails_closed(attack):
    report, manifest, request = example()
    native = report['native_production_comparison']
    if attack == 'missing': report.pop('native_production_comparison')
    elif attack == 'old_request': native['request']['request_id'] = 'stale'
    elif attack == 'wrong_source': native['source_hashes'] = {'source/kernel.py': 'e' * 64}
    elif attack == 'wrong_image': native['runtime_image'] = 'another image'
    elif attack == 'wrong_provenance': native['native_source_manifest_sha256'] = 'e' * 64
    elif attack == 'missing_case': native['cases'].pop()
    elif attack == 'duplicate_case': native['cases'][1] = copy.deepcopy(native['cases'][0])
    elif attack == 'changed_shape': native['cases'][0]['legs']['candidate_port']['case']['tensors']['A']['shape'] = [1]
    elif attack == 'changed_count': native['cases'][0]['legs']['native_production']['case']['occurrences'] = 1
    elif attack == 'missing_sample': native['cases'][0]['legs']['native_production']['samples_ms'].pop()
    elif attack == 'wrong_method': native['cases'][0]['legs']['native_production']['benchmark_method'] = 'host'
    elif attack == 'missing_check': native['cases'][0]['legs']['candidate_port']['oracle_checks'] = 1
    elif attack == 'unmatched_inputs': native['cases'][0]['identical_captured_ABI_and_fresh_numeric_challenge_sequence'] = False
    elif attack == 'unqualified_native': native['cases'][0]['native_output_parity'] = False
    elif attack == 'diagnostic_only': native['diagnostic_only'] = True
    with pytest.raises(ValueError):
        validate_native_measurements(report, manifest, request, request['source_sha256'], 'd' * 64)


def test_secondary_comparison_requires_the_same_complete_frozen_contract():
    before = as_test_cases(measured(), is_baseline=True)
    after = as_test_cases(measured(port_ms=2, candidate_ms=2))
    with pytest.raises(ValueError, match='complete'):
        metric_summary(before[:-1], after)
    after[0].metadata['native_source_manifest_sha256'] = 'e' * 64
    with pytest.raises(ValueError, match='contracts differ'):
        metric_summary(before, after)


def test_normal_arena_result_and_saved_series_use_native_primary(tmp_path, monkeypatch):
    before = as_test_cases(measured(), is_baseline=True)
    after = as_test_cases(measured(source_hash='e' * 64, request_id='optimized', port_ms=2, candidate_ms=2))
    monkeypatch.setattr(evaluator, 'evaluate_compilation', lambda *a, **k: (True, None))
    monkeypatch.setattr(evaluator, 'evaluate_correctness', lambda *a, **k: (True, None))
    monkeypatch.setattr(evaluator, 'measure_performance', lambda *a, **k: after)
    config = {'task_type': 'instruction2triton', 'scoring_baseline': POLICY,
              'trusted_evaluation': {'schema_version': 1}}
    result = evaluator.evaluate_kernel(tmp_path, config, before)
    evaluator.write_task_result(tmp_path, result, before, 'ported-gemm', 'test', create_plots=False)
    saved = yaml.safe_load((tmp_path / 'task_result.yaml').read_text())
    assert saved['baseline_kind'] == 'native_production'
    assert saved['base_execution_time'] == 1 and saved['best_optimized_execution_time'] == 2
    assert saved['speedup_ratio'] == 0.5 and saved['port_to_port_speedup_ratio'] == 2
    assert saved['production_kernel_improvement'] is False
    for name in ('build/initial_native_baseline_perf.yaml', 'baseline_perf.yaml',
                 'build/port_baseline_perf.yaml', 'build/port_optimized_perf.yaml'):
        assert (tmp_path / name).is_file()
    from src.testcases import load_performance_results
    restored = load_performance_results(tmp_path / 'build', 'initial_native_baseline_perf.yaml')
    assert metric_summary(restored, after)['port_to_port_speedup_ratio'] == 2
    assert restored[0].metadata['native_comparison_challenge_seed'] == 731


def test_native_evidence_is_mandatory_in_normal_performance_path(tmp_path, monkeypatch):
    (tmp_path / 'source').mkdir(); (tmp_path / 'source/kernel.py').write_text('GPU source')
    (tmp_path / 'provenance').mkdir(); (tmp_path / 'provenance/NATIVE.json').write_text('pinned native source')
    (tmp_path / 'build').mkdir()
    source_hash = hashlib.sha256((tmp_path / 'source/kernel.py').read_bytes()).hexdigest()
    report, manifest, request = example(source_hash=source_hash)
    report['native_production_comparison']['native_source_manifest_sha256'] = hashlib.sha256(
        (tmp_path / 'provenance/NATIVE.json').read_bytes()).hexdigest()
    (tmp_path / 'cases.json').write_text(json.dumps(manifest))
    config = {'task_type': 'instruction2triton', 'performance_command': ['benchmark'],
              'source_file_path': ['source/kernel.py'], 'scoring_baseline': POLICY,
              'trusted_evaluation': {'schema_version': 1, 'case_manifest': 'cases.json'}}
    monkeypatch.setattr(performance, 'force_jit_rebuild', lambda *a: {})
    def run(*args, **kwargs):
        assert not (tmp_path / 'build/performance_report.json').exists()
        (tmp_path / 'build/performance_report.json').write_text(json.dumps(report))
        return True, 'Performance: 0.000001 ms', ''
    monkeypatch.setattr(performance, 'run_command', run)
    assert performance.measure_performance(tmp_path, config)[0].execution_time_ms == 4
    assert performance.measure_baseline(tmp_path, config)[0].execution_time_ms == 1
    report.pop('native_production_comparison')
    assert performance.measure_performance(tmp_path, config) == []


def test_native_scoring_remains_explicit_opt_in():
    assert scoring_policy({}) is None
    with pytest.raises(ValueError):
        scoring_policy({'scoring_baseline': {**POLICY, 'kind': 'choose_by_source_hash'},
                        'trusted_evaluation': {'schema_version': 1}})


PORT_TASKS = ['glm-5.3-flash__fused_moe_kernel', 'glm-5.3-flash__ck_gemm_a8w8_blockscale_bpreshuffle',
              'glm-5.3-flash__gemm_a16w16_bf16_cijk', 'kimi-k3__dense_bf16_gemm_cijk']


@pytest.mark.parametrize('name', PORT_TASKS)
def test_actual_comparator_report_producers_bind_scoreable_evidence(tmp_path, name):
    """Execute each real producer's report-finalization code after CPU sample stand-ins."""
    import ast
    from types import SimpleNamespace
    task = Path(__file__).resolve().parents[1] / 'tasks/headkernel' / name
    tree = ast.parse((task / 'scripts/production_comparison.py').read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == ('compare' if name == 'glm-5.3-flash__gemm_a16w16_bf16_cijk' else 'main'))
    start = next(i for i, node in enumerate(function.body) if isinstance(node, ast.If)
                 and 'Task changed during native comparison' in ast.unparse(node))
    assert 'request' in [arg.arg for arg in function.args.args]
    (tmp_path / 'build').mkdir(); (tmp_path / 'provenance').mkdir(); (tmp_path / 'source').mkdir()
    (tmp_path / 'source/kernels.py').write_text('GPU source')
    source_hash = hashlib.sha256((tmp_path / 'source/kernels.py').read_bytes()).hexdigest()
    report, manifest, request = example(source_hash=source_hash)
    request['source_sha256'] = {'source/kernels.py': source_hash}
    comparisons = report.pop('native_production_comparison')['cases']
    report['request'] = request
    (tmp_path / 'build/performance_report.json').write_text(json.dumps(report))
    native_path = 'provenance/NATIVE-SOURCES.json' if name == PORT_TASKS[0] else 'provenance/NATIVE-BASELINE.json'
    (tmp_path / native_path).write_text('{"source": "pinned"}\n')
    receipts = tmp_path / 'build/receipt.jsonl'; receipts.write_text('{}\n')
    fake_task = SimpleNamespace(ROOT=tmp_path, source_hash=lambda: source_hash,
        package_hash=lambda: 'unchanged', strict_json=json.loads, fingerprint=fingerprint,
        file_sha=lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest())
    namespace = {'task': fake_task, 'request': request, 'manifest': manifest, 'comparisons': comparisons,
                 'before': 'unchanged', 'hashlib': hashlib, 'json': json, 'challenge': 731,
                 'receipts': SimpleNamespace(path=receipts)}
    exec(compile(ast.Module(body=function.body[start:], type_ignores=[]), '<actual-native-report-producer>', 'exec'), namespace)
    config = {'source_file_path': ['source/kernels.py'], 'trusted_evaluation': {'schema_version': 1},
              'scoring_baseline': {**POLICY, 'native_source_manifest': native_path}}
    (tmp_path / 'cases.json').write_text(json.dumps(manifest))
    measured = load_native_measurements(tmp_path, config, request=request)
    assert measured['native'][0]['execution_time_ms'] == 1
    assert measured['candidate'][0]['execution_time_ms'] == 4
    raw = json.loads((tmp_path / 'build/native_production_comparison.json').read_text())
    assert raw['challenge_seed'] == measured['comparison_challenge_seed'] == 731
    assert raw['request']['challenge_seed'] == 42


@pytest.mark.parametrize('invalid_seed', [None, True, -1, '731'])
def test_private_comparison_seed_is_required_and_not_replaced_by_parent_seed(invalid_seed):
    report, manifest, request = example()
    report['native_production_comparison']['challenge_seed'] = invalid_seed
    with pytest.raises(ValueError, match='actual private challenge seed'):
        validate_native_measurements(report, manifest, request, request['source_sha256'], 'd' * 64)


def test_dense_failure_wrapper_forwards_the_enclosing_request():
    import ast
    task = Path(__file__).resolve().parents[1] / 'tasks/headkernel' / PORT_TASKS[2]
    tree = ast.parse((task / 'scripts/production_comparison.py').read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    seen = []
    namespace = {'compare': lambda request: seen.append(request)}
    exec(compile(ast.Module(body=[main], type_ignores=[]), '<actual-comparator-wrapper>', 'exec'), namespace)
    request = {'request_id': 'enclosing-performance'}
    namespace['main'](request=request)
    assert seen == [request]
