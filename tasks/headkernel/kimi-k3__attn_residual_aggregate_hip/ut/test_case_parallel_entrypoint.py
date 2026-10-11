"""Exercise full-task finalization and internal worker isolation without GPUs."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from evaluation_contract import fingerprint


TASK = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('_kimi_parallel_entrypoint_test', TASK / 'scripts/task_runner.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def inputs(root, phase):
    full = json.loads((TASK / 'cases.json').read_text())
    (root / 'cases.json').write_text(json.dumps(full))
    request = {'schema_version': 1, 'phase': phase, 'request_id': 'parent-request',
               'manifest_sha256': fingerprint(full), 'package_sha256': 'original-package',
               'source_sha256': {'source/attn_res_hip.py': 'original-source'}, 'challenge_seed': 19}
    return full, request


def payload(full, request, phase, selected=None):
    cases = full['cases'] if selected is None else selected
    receipts = [{'case_id': case['case_id'], 'source_sha256': request['source_sha256'],
                 'candidate_invoked': True, 'independent_reference_invoked': True} for case in cases]
    rows = []
    for case in cases:
        row = {'case': case, 'correct': True}
        if phase == 'correctness':
            row.update(seeds=[0, 1, 2], negative_controls={'no_op': True, 'wrong_output': True})
        elif phase == 'performance':
            row.update(samples_ms=[0.25] * 100, benchmark_method='cuda_graph', warmup_iterations=10,
                       fresh_input_resets=100, output_initializations=100, oracle_checks=100)
        else:
            continue
        rows.append(row)
    return {'compiled': True, 'compiled_specializations': receipts, 'cases': rows,
            'oracle_order': 'CPU input truth before candidate; CPU output and post-input snapshots before reference'}


def main_context(root, request, helper, phase, worker=False):
    argv = ['task_runner.py', phase, '--request', 'parent.json']
    if worker:
        argv += ['--case-worker', 'assignment.json']
    return (patch.object(runner, 'ROOT', root), patch.object(runner, 'validate_sources'),
            patch.object(runner, 'request_for', return_value=request),
            patch.dict(sys.modules, case_parallel=helper, torch=None),
            patch.object(sys, 'argv', argv))


@pytest.mark.parametrize('assignment', ['assignment.json', ''])
def test_worker_flag_requires_a_parent_request_before_touching_outputs(tmp_path, assignment):
    with patch.object(runner, 'ROOT', tmp_path), \
            patch.object(sys, 'argv', ['task_runner.py', 'performance', '--case-worker', assignment]), \
            pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2 and not (tmp_path / 'build').exists()


@pytest.mark.parametrize('phase', ['compile', 'correctness', 'performance'])
def test_parallel_parent_finalizes_one_complete_task_without_GPU_initialization(tmp_path, monkeypatch, phase):
    monkeypatch.setenv('AKA_CASE_GPU_PLAN', '/read-only/plan.json')
    full, original = inputs(tmp_path, phase)
    unchanged = copy.deepcopy(original)
    calls = []
    plan = {'assignment': 'full-domain'}
    proof = {'schema_version': 1, 'plan_sha256': 'plan-hash', 'assignment_sha256': 'assignment-hash'}

    def bind(request, selected_plan):
        calls.append('bind')
        assert request == unchanged and selected_plan is plan
        return {**request, 'case_parallelism': proof, 'gpu_group': [{'worker_id': 0}]}

    def workers(root, manifest, request, selected_plan, entrypoint, *, leg):
        calls.append('workers')
        assert root == tmp_path and manifest == full and leg == 'framework'
        assert request['case_parallelism'] == proof and request['challenge_seed'] == 19
        data = payload(full, request, phase)
        return {'rows': data['cases'], 'compiled_specializations': data['compiled_specializations'],
                'workers': [{'worker_id': 0}], 'case_parallelism': {**proof, 'workers': [{'worker_id': 0}]}}

    helper = SimpleNamespace(load_plan=lambda *args: plan, bind_request=bind, run_workers=workers)
    contexts = main_context(tmp_path, original, helper, phase)
    with contexts[0], contexts[1], contexts[2], contexts[3], contexts[4], \
            patch.object(runner, 'runtime_on_gpu', side_effect=AssertionError('parent touched GPU')):
        runner.main()
    assert calls == ['bind', 'workers'] and original == unchanged
    report = json.loads((tmp_path / 'build' / (phase + '_report.json')).read_text())
    assert report['status'] == 'ok' and report['case_parallelism']['workers'] == [{'worker_id': 0}]
    assert len(report['compiled_specializations']) == 69
    assert len(report['cases']) == (0 if phase == 'compile' else 69)
    if phase == 'performance':
        assert len(report['test_cases']) == 69
        assert all(len(row['samples_ms']) == 100 for row in report['cases'])


@pytest.mark.parametrize('kind', ['missing_compile', 'duplicate_compile', 'wrong_source', 'missing_row',
                                  'short_samples', 'long_samples', 'missing_seed', 'missing_control'])
def test_parallel_parent_rejects_incomplete_or_invalid_union(tmp_path, monkeypatch, kind):
    monkeypatch.setenv('AKA_CASE_GPU_PLAN', '/read-only/plan.json')
    phase = 'compile' if 'compile' in kind else 'correctness' if kind in ('missing_seed', 'missing_control') else 'performance'
    full, request = inputs(tmp_path, phase)
    data = payload(full, request, phase)
    if kind == 'missing_compile':
        data['compiled_specializations'].pop()
    elif kind == 'duplicate_compile':
        data['compiled_specializations'][-1] = data['compiled_specializations'][0]
    elif kind == 'wrong_source':
        data['compiled_specializations'][0]['source_sha256'] = {'source/attn_res_hip.py': 'foreign'}
    elif kind == 'missing_row':
        data['cases'].pop()
    elif kind == 'short_samples':
        data['cases'][0]['samples_ms'].pop()
    elif kind == 'long_samples':
        data['cases'][0]['samples_ms'].append(0.25)
    elif kind == 'missing_seed':
        data['cases'][0]['seeds'].pop()
    else:
        data['cases'][0]['negative_controls'].pop('wrong_output')
    combined = {'rows': data['cases'], 'compiled_specializations': data['compiled_specializations'],
                'case_parallelism': {'schema_version': 1}}
    helper = SimpleNamespace(load_plan=lambda *args: {}, bind_request=lambda request, plan: request,
                             run_workers=lambda *args, **kwargs: combined)
    build = tmp_path / 'build'
    build.mkdir()
    report = build / (phase + '_report.json')
    report.write_text('stale report')
    report.with_suffix('.tmp').write_text('stale temporary')
    contexts = main_context(tmp_path, request, helper, phase)
    with contexts[0], contexts[1], contexts[2], contexts[3], contexts[4], \
            patch.object(runner, 'runtime_on_gpu', side_effect=AssertionError('parent touched GPU')), \
            pytest.raises(ValueError):
        runner.main()
    assert not report.exists() and not report.with_suffix('.tmp').exists()


@pytest.mark.parametrize('phase', ['compile', 'correctness', 'performance'])
def test_internal_worker_preserves_parent_request_and_writes_only_private_receipt(tmp_path, monkeypatch, phase):
    monkeypatch.setenv('AKA_CASE_GPU_PLAN', '/read-only/plan.json')
    full, request = inputs(tmp_path, phase)
    request.update(case_parallelism={'schema_version': 1}, gpu_group=[{'worker_id': 0}])
    original = copy.deepcopy(request)
    selected = [full['cases'][index] for index in (0, 1, 3)]
    private = tmp_path / 'build/case-parallel/parent-request/worker-0'
    context = SimpleNamespace(cases=selected, output_directory=private)
    calls = []

    def enter(root, manifest, parent, assignment):
        calls.append('preflight')
        assert parent == original and assignment == 'assignment.json'
        return context

    def runtime(manifest, parent, *, build):
        calls.append('runtime')
        assert calls[:2] == ['preflight', 'runtime'] and build == private and parent == original
        return SimpleNamespace(cpu_thread_policy={'intraop_after': 8}), object()

    def evaluate(runtime, manifest, phase, parent, torch, *, case_ids, progress):
        calls.append('evaluate')
        assert case_ids == [case['case_id'] for case in selected] and parent == original
        progress({'phase': phase, 'case_count': 3, 'full_case_count': 69, 'cases_completed': 3}, False)
        return payload(full, parent, phase, selected)

    def finish(actual_context, **values):
        calls.append('finish')
        assert actual_context is context
        assert len(values['compiled_specializations']) == 3
        assert values['evidence']['completed_case_ids'] == [case['case_id'] for case in selected]
        (private / 'worker-result.json').write_text(json.dumps({'status': 'partition_complete', **values}))

    helper = SimpleNamespace(load_plan=lambda *args: {}, enter_worker=enter, finish_worker=finish,
                             bind_request=lambda *args: pytest.fail('worker rebound the parent request'))
    (tmp_path / 'build').mkdir()
    public = tmp_path / 'build' / (phase + '_report.json')
    public.write_text('parent-owned report')
    contexts = main_context(tmp_path, request, helper, phase, worker=True)
    with contexts[0], contexts[1], contexts[2], contexts[3], contexts[4], \
            patch.object(runner, 'runtime_on_gpu', runtime), patch.object(runner, 'evaluate_cases', evaluate), \
            patch.object(runner, 'finalize_report', side_effect=AssertionError('worker finalized a task')):
        runner.main()
    assert calls == ['preflight', 'runtime', 'evaluate', 'finish'] and request == original
    assert public.read_text() == 'parent-owned report'
    assert (private / 'worker-result.json').is_file() and (private / (phase + '_progress.json')).is_file()
    assert not (tmp_path / 'build' / (phase + '_progress.json')).exists()


@pytest.mark.parametrize('failure', ['preflight', 'oracle'])
def test_failed_worker_never_overwrites_or_removes_parent_report(tmp_path, monkeypatch, failure):
    monkeypatch.setenv('AKA_CASE_GPU_PLAN', '/read-only/plan.json')
    full, request = inputs(tmp_path, 'performance')
    private = tmp_path / 'build/worker-0'

    def enter(*args):
        if failure == 'preflight':
            raise ValueError('wrong GPU proof')
        return SimpleNamespace(cases=full['cases'][:1], output_directory=private)

    helper = SimpleNamespace(load_plan=lambda *args: {}, enter_worker=enter,
                             finish_worker=lambda *args, **kwargs: pytest.fail('failed worker completed'))
    (tmp_path / 'build').mkdir()
    public = tmp_path / 'build/performance_report.json'
    public.write_text('parent-owned report')
    contexts = main_context(tmp_path, request, helper, 'performance', worker=True)
    with contexts[0], contexts[1], contexts[2], contexts[3], contexts[4], \
            patch.object(runner, 'runtime_on_gpu', return_value=(object(), object())), \
            patch.object(runner, 'evaluate_cases', side_effect=ValueError('oracle failure')), \
            pytest.raises(ValueError):
        runner.main()
    assert public.read_text() == 'parent-owned report'
    assert not (private / 'worker-result.json').exists()
