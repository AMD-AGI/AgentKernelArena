"""CPU evidence for SIKL protocol/lifecycle and numerical guard logic, not GPU validation."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import types

import pytest
import yaml

from src.task_protocol import (CaseManifest, baseline_correctness_accepted,
                               parse_command_result)
from src.task_spec import load_task_spec

ROOT = Path(__file__).resolve().parents[1]
TASKS = sorted((ROOT / 'tasks/SIKL-task').glob('*/config.yaml'))
REPRESENTATIVES = ['gemm_a16w16_nt_n32_k6144', 'mxfp4_moe_e65_i1024']
MODULE_NAMES = ['task_contract', 'task_inputs', 'task_compare', 'task_initialize',
                'task_reference', 'task_baseline', 'task_measure', 'evaluate', 'export_solution']


@contextmanager
def modules(task, monkeypatch):
    # Bare imports are deliberately task-local in production, with a separate
    # subprocess per action. Isolate them equally in these in-process tests.
    with monkeypatch.context() as patch:
        patch.syspath_prepend(str(task / 'scripts'))
        for name in MODULE_NAMES:
            patch.delitem(sys.modules, name, raising=False)
        contract = importlib.import_module('task_contract')
        yield contract
        for name in MODULE_NAMES:
            sys.modules.pop(name, None)


def parse_report(report):
    return parse_command_result('ARENA_EVAL_RESULT=' + json.dumps(report, allow_nan=False),
                                role=report['role'], action=report['action'],
                                returncode=0 if report['status'] == 'PASS' else 1)


def manifest(contract):
    return CaseManifest.from_result(parse_report({
        'protocol': 'arena-eval-v1', 'role': 'task', 'action': 'validate-task',
        'status': 'PASS', 'cases': contract.case_manifest(contract.load_workload()),
        'metadata': {'candidate_state': 'unimplemented'},
    }))


@pytest.mark.parametrize('config_path', TASKS, ids=lambda p: p.parent.name)
def test_complete_manifest_is_accepted_by_framework(config_path, monkeypatch):
    with modules(config_path.parent, monkeypatch) as contract:
        captured = manifest(contract)
        assert len(captured.cases) == 13
        for row in captured.cases:
            assert row['checks'] == ['correctness', 'performance']
            assert row['params']['uuid']
            assert row['dtype'] == 'bfloat16'


@pytest.mark.parametrize('config_path', TASKS, ids=lambda p: p.parent.name)
@pytest.mark.parametrize('action', ['compile', 'correctness', 'performance'])
def test_real_cli_stub_fails_with_complete_case_evidence(config_path, action, monkeypatch):
    task = config_path.parent
    result = subprocess.run([sys.executable, 'scripts/evaluate.py', 'candidate', action],
                            cwd=task, text=True, capture_output=True, timeout=30)
    assert result.returncode == 1
    report = parse_command_result(result.stdout, role='candidate', action=action, returncode=1)
    with modules(task, monkeypatch) as contract:
        manifest(contract).validate(report)
    assert not report.passed
    assert all('unimplemented' in row['reason'] for row in report.cases)
    assert all(row['failure_kind'] != 'numerical_mismatch' for row in report.cases)


@pytest.mark.parametrize('name', REPRESENTATIVES)
@pytest.mark.parametrize('source', [
    'import task_reference', 'import task_baseline as baseline',
    'from scripts import task_reference', 'from scripts import task_baseline as base',
    'from scripts.task_reference import run', 'from scripts import *',
    'from .scripts import task_baseline', 'from . import task_reference',
    'from aiter import tuned_gemm', 'import aiter.fused_moe',
    'from torch import matmul as product', 'import importlib',
    'def f(a, b):\n return a @ b.T', 'def f(a,b):\n return torch.mm(a,b)',
    '__import__("task_reference")', 'open("scripts/task_reference.py")',
])
def test_import_members_aliases_and_direct_baseline_paths_rejected(name, source, monkeypatch):
    with modules(ROOT / 'tasks/SIKL-task' / name, monkeypatch) as contract:
        with pytest.raises(RuntimeError):
            contract.assert_source_independent(source)


def test_allowed_host_plumbing_and_flydsl_imports(monkeypatch):
    with modules(ROOT / 'tasks/SIKL-task' / REPRESENTATIVES[0], monkeypatch) as contract:
        contract.assert_source_independent('from functools import lru_cache\nimport torch\nimport flydsl.compiler as flyc\n@lru_cache(None)\ndef build(**axes):\n return lambda a,b: torch.empty_like(a)\n')


@pytest.mark.parametrize('name', REPRESENTATIVES)
def test_initial_state_is_examined_not_echoed(name, tmp_path, monkeypatch):
    task = tmp_path / 'task'
    shutil.copytree(ROOT / 'tasks/SIKL-task' / name, task)
    with modules(task, monkeypatch) as contract:
        runner = importlib.import_module('evaluate')
        config = contract.load_config()
        monkeypatch.setenv('ARENA_EVAL_PHASE', 'task_validation')
        assert runner.check_initial_state(config) == 'unimplemented'
        (task / 'kernel.py').unlink()
        assert runner.check_initial_state(config) == 'unimplemented'
        entry = contract.candidate_entry(config)
        (task / entry['file']).write_text(f'def {entry["symbol"]}(**kw):\n return lambda *args: None\n')
        with pytest.raises(RuntimeError, match='declared unimplemented'):
            runner.check_initial_state(config)
        monkeypatch.setenv('ARENA_EVAL_PHASE', 'candidate_evaluation')
        assert runner.check_initial_state(config) == 'implemented'
        (task / entry['file']).write_text('raise RuntimeError("broken skeleton")')
        with pytest.raises(RuntimeError, match='executable'):
            runner.check_initial_state(config)


@pytest.mark.parametrize('name', REPRESENTATIVES)
def test_shape_and_numerical_failures_are_distinct(name, monkeypatch):
    torch = pytest.importorskip('torch')
    with modules(ROOT / 'tasks/SIKL-task' / name, monkeypatch):
        measure = importlib.import_module('task_measure')
        expected = torch.ones((2, 3), dtype=torch.bfloat16)
        assert measure.compare_output(expected.clone(), expected)['status'] == 'PASS'
        wrong = measure.compare_output(torch.zeros_like(expected), expected)
        assert wrong['failure_kind'] == 'numerical_mismatch'
        assert wrong['metadata']['output_contract_passed'] is True
        for got in [torch.zeros(1), expected.float(), expected * float('nan'), None]:
            assert measure.compare_output(got, expected)['failure_kind'] == 'output_contract'
        with pytest.raises(ValueError, match='reference'):
            measure.compare_output(expected, expected * float('inf'))


@pytest.mark.parametrize('name', REPRESENTATIVES)
def test_changed_but_wrong_timed_output_is_rejected(name, monkeypatch):
    torch = pytest.importorskip('torch')
    with modules(ROOT / 'tasks/SIKL-task' / name, monkeypatch):
        measure = importlib.import_module('task_measure')
        monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
        inputs = {'a': torch.tensor([[1., 2.]], dtype=torch.bfloat16)}
        output = torch.zeros((1, 2), dtype=torch.bfloat16)
        monkeypatch.setattr(measure.task_inputs, 'refill_case_inputs', lambda i: i['a'].fill_(2))
        monkeypatch.setattr(measure.task_inputs, 'call_kwargs', lambda i: i)
        monkeypatch.setattr(measure.task_reference, 'run', lambda **kw: torch.tensor([[18., 28.]], dtype=torch.bfloat16))
        timed = types.SimpleNamespace(bound=True, outputs=output,
                                     rerun=lambda: output.copy_(inputs['a']))
        result = measure.verify_timed_invocation(inputs, timed)
        assert output.tolist() == [[2., 2.]]  # Finite and different, still wrong.
        assert result['failure_kind'] == 'numerical_mismatch'
        assert result['metadata']['replay_checked']
        timed.rerun = lambda: output.copy_(torch.tensor([[18., 28.]], dtype=torch.bfloat16))
        output.zero_()
        assert measure.verify_timed_invocation(inputs, timed)['status'] == 'PASS'
        timed.rerun = lambda: output  # Poison survives an empty replay.
        assert measure.verify_timed_invocation(inputs, timed)['failure_kind'] == 'output_contract'


@pytest.mark.parametrize('name', REPRESENTATIVES)
def test_input_modification_rejected_and_missing_launch_never_falls_back(name, monkeypatch):
    torch = pytest.importorskip('torch')
    with modules(ROOT / 'tasks/SIKL-task' / name, monkeypatch):
        measure = importlib.import_module('task_measure')
        inputs = {'a': torch.ones(2, 3)}
        snapshot = measure.input_snapshot(inputs)
        inputs['a'].zero_()
        with pytest.raises(RuntimeError, match='protected input'):
            measure.assert_inputs_unchanged(inputs, snapshot)
        with pytest.raises(RuntimeError, match='no baseline fallback'):
            measure.case_call(inputs, role='candidate')
        with pytest.raises(NotImplementedError):
            measure.build_launch(lambda **kw: (_ for _ in ()).throw(NotImplementedError()), measure.task_inputs.CASES[0])


def test_all_seven_actions_use_full_identity_and_explicit_role_with_injected_cpu_backend(tmp_path, monkeypatch):
    """The injected execution backend tests command orchestration, not GPU kernels."""
    pytest.importorskip('torch')
    task = tmp_path / 'task'
    shutil.copytree(ROOT / 'tasks/SIKL-task' / REPRESENTATIVES[0], task)
    with modules(task, monkeypatch) as contract:
        runner = importlib.import_module('evaluate')
        measure = importlib.import_module('task_measure')
        monkeypatch.setenv('ARENA_EVAL_PHASE', 'task_validation')
        # Model the framework's materialized baseline source.
        path = task / contract.load_config()['baseline']['source_files'][0]
        path.parent.mkdir(parents=True)
        path.write_text('# explanatory material')
        monkeypatch.setattr(runner, 'require_runtime', lambda w: None)
        monkeypatch.setattr(runner, 'runtime_evidence', lambda w: {'backend': 'injected CPU unit test'})
        monkeypatch.setattr(runner, 'dispatch_evidence', lambda w, c: {})
        monkeypatch.setattr(runner, 'validate_case', lambda c, m: {'status': 'PASS'})
        validated = parse_report(runner.run('task', 'validate-task'))
        captured = CaseManifest.from_result(validated)
        assert validated.metadata['candidate_state'] == 'unimplemented'
        monkeypatch.setattr(runner, 'load_candidate', lambda c: lambda **kw: lambda *a: None)
        called = []
        def compiled(c, *, role, launch, measure):
            called.append((role, c['case_id'], callable(launch)))
            return {'status': 'PASS'}
        monkeypatch.setattr(runner, 'compile_case', compiled)
        def checked(c, *, role, launch):
            if role == 'baseline':
                return {'status': 'FAIL', 'failure_kind': 'numerical_mismatch', 'reason': 'injected finite mismatch'}
            return {'status': 'PASS'}
        monkeypatch.setattr(measure, 'check_case', checked)
        monkeypatch.setattr(measure, 'time_case', lambda *a, **kw: {'status': 'PASS', 'execution_time_ms': 0.01, 'benchmark_method': 'cuda_graph'})
        spec = load_task_spec(task / 'config.yaml', task_id='SIKL-task/unit-test')
        for role in ('baseline', 'candidate'):
            for action in ('compile', 'correctness', 'performance'):
                report = parse_report(runner.run(role, action))
                captured.validate(report)
                if (role, action) == ('baseline', 'correctness'):
                    assert baseline_correctness_accepted(report, baseline=spec.baseline, phase='task_validation', manifest=captured)
                    assert not baseline_correctness_accepted(report, baseline=spec.baseline, phase='candidate_evaluation', manifest=captured)
                else:
                    assert report.passed
        assert len(called) == 26
        assert all(has_launch == (role == 'candidate') for role, _, has_launch in called)
        monkeypatch.setattr(measure, 'check_case', lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('injected crash')))
        crashed = parse_report(runner.run('baseline', 'correctness'))
        assert len(crashed.cases) == 13
        assert not baseline_correctness_accepted(crashed, baseline=spec.baseline, phase='task_validation', manifest=captured)


def test_baseline_replay_diagnostic_cannot_exempt_candidate_or_non_numerical_errors(monkeypatch):
    pytest.importorskip('torch')
    with modules(ROOT / 'tasks/SIKL-task' / REPRESENTATIVES[0], monkeypatch):
        measure = importlib.import_module('task_measure')
        helper = types.SimpleNamespace(TimedRun=lambda: object(), benchmark_cuda_graph_or_events=lambda *a, **kw: (0.1, {'benchmark_method': 'cuda_graph'}))
        monkeypatch.setitem(sys.modules, '_aka_benchmark', helper)
        monkeypatch.setattr(measure.task_inputs, 'build_case_inputs', lambda c: {})
        monkeypatch.setattr(measure, 'case_call', lambda *a, **kw: lambda: None)
        mismatch = {'status': 'FAIL', 'failure_kind': 'numerical_mismatch', 'reason': 'finite mismatch'}
        monkeypatch.setattr(measure, 'verify_timed_invocation', lambda *a: mismatch)
        baseline = measure.time_case({}, role='baseline', baseline_diagnostic=True)
        assert baseline['status'] == 'PASS'
        assert baseline['metadata']['replay_correctness']['status'] == 'FAIL'
        assert measure.time_case({}, role='candidate', launch=lambda: None, baseline_diagnostic=True)['status'] == 'FAIL'
        assert measure.time_case({}, role='baseline', baseline_diagnostic=False)['status'] == 'FAIL'
        mismatch['failure_kind'] = 'output_contract'
        assert measure.time_case({}, role='baseline', baseline_diagnostic=True)['status'] == 'FAIL'


def test_declared_nested_paths_builder_identity_and_export_are_honored(tmp_path, monkeypatch):
    task = tmp_path / 'task'
    shutil.copytree(ROOT / 'tasks/SIKL-task' / REPRESENTATIVES[0], task)
    config = yaml.safe_load((task / 'config.yaml').read_text())
    config['candidate']['editable'] = ['source/generated.py']
    config['candidate']['entrypoints'][0].update(file='source/generated.py', symbol='build_unrelated_name')
    config['evaluation']['workloads'] = 'data/cases.json'
    config['exports'][0]['output'] = 'result/accepted.json'
    (task / 'source').mkdir()
    (task / 'source/generated.py').write_text('def build_unrelated_name(**kw):\n return lambda a,b: a\n')
    (task / 'data').mkdir()
    (task / 'workload.json').rename(task / 'data/cases.json')
    (task / 'config.yaml').write_text(yaml.safe_dump(config))
    accepted = dict(pass_compilation=True, pass_correctness=True, pass_tool_gate=True,
                    workload_consistent=True, benchmark_method_consistent=True,
                    valid_baseline_cases=13, valid_optimized_cases=13,
                    best_optimized_execution_time=0.01)
    (task / 'task_result.yaml').write_text(yaml.safe_dump(accepted))
    with modules(task, monkeypatch) as contract:
        runner = importlib.import_module('evaluate')
        assert runner.load_candidate(config)(m=1)('a', 'b') == 'a'
        assert len(contract.load_workload()['cases']) == 13
        exporter = importlib.import_module('export_solution')
        exporter.export()
        artifact = json.loads((task / 'result/accepted.json').read_text())
        assert artifact['sources'][0]['path'] == 'source/generated.py'
        assert 'build_unrelated_name' in artifact['sources'][1]['content']
        assert artifact['spec']['entry_point'] == 'sikl_entry.py::run'
        assert json.loads((task / 'solution.json').read_text())['spec']['entry_point'] == ''
        assert yaml.safe_load((task / 'task_result.yaml').read_text()) == accepted
        for key, value in [('pass_correctness', False), ('pass_tool_gate', False),
                           ('valid_optimized_cases', 12), ('best_optimized_execution_time', 0)]:
            with pytest.raises(ValueError):
                exporter.accepted_result({**accepted, key: value}, 13)
        for path in ['../escape.py', '/tmp/escape.py', 'a/../../b', 'C:/escape.py']:
            with pytest.raises(ValueError):
                contract.task_path(path, must_exist=False)
        (task / 'escape').symlink_to(tmp_path)
        with pytest.raises(ValueError):
            contract.task_path('escape/out.py', must_exist=False)


def test_nonfinite_diagnostics_are_valid_json_without_changing_verdict(monkeypatch):
    with modules(ROOT / 'tasks/SIKL-task' / REPRESENTATIVES[1], monkeypatch) as contract:
        original = {'status': 'PASS', 'metrics': {'sqnr_db': float('inf')},
                    'metadata': {'sqnr_db_nonfinite': 'positive infinity for exact match'}}
        safe = contract.json_safe(original)
        assert safe['status'] == 'PASS'
        assert safe['metrics']['sqnr_db'] is None
        assert 'positive infinity' in safe['metadata']['sqnr_db_nonfinite']
        json.dumps(safe, allow_nan=False)


@pytest.mark.parametrize('name', REPRESENTATIVES)
def test_reference_callback_does_not_accept_invalid_reference_or_nonfinite_output(name, monkeypatch):
    torch = pytest.importorskip('torch')
    with modules(ROOT / 'tasks/SIKL-task' / name, monkeypatch) as contract:
        measure = importlib.import_module('task_measure')
        expected = torch.zeros((2, 2), dtype=torch.bfloat16)
        exact = measure.compare_output(expected.clone(), expected)
        assert exact['status'] == 'PASS'
        if name.startswith('mxfp4'):
            assert exact['metrics']['sqnr_db'] is None
            assert 'exact match' in exact['metadata']['sqnr_db_nonfinite']
        json.dumps(contract.json_safe(exact), allow_nan=False)


@pytest.mark.parametrize('mutation', ['duplicate_id', 'duplicate_uuid', 'invalid_axis', 'invalid_duration', 'empty_cases'])
def test_manifest_rejects_malformed_task_data(mutation, monkeypatch):
    with modules(ROOT / 'tasks/SIKL-task' / REPRESENTATIVES[0], monkeypatch) as contract:
        workload = deepcopy(contract.load_workload())
        if mutation == 'duplicate_id':
            workload['cases'][1]['case_id'] = workload['cases'][0]['case_id']
        elif mutation == 'duplicate_uuid':
            workload['cases'][1]['uuid'] = workload['cases'][0]['uuid']
        elif mutation == 'invalid_axis':
            workload['axes']['n'] = 0
        elif mutation == 'invalid_duration':
            workload['bench']['target_ms'] = float('inf')
        else:
            workload['cases'] = []
        with pytest.raises(ValueError):
            contract.case_manifest(workload)


@pytest.mark.parametrize('name', REPRESENTATIVES)
def test_exported_tensor_binding_executes_declared_builder(tmp_path, monkeypatch, name):
    """Exercise the binding with host sentinels, not a claim of FlyDSL execution."""
    task = tmp_path / 'task'
    shutil.copytree(ROOT / 'tasks/SIKL-task' / name, task)
    artifact_dir = tmp_path / 'unpacked'
    artifact_dir.mkdir()
    (artifact_dir / 'nested').mkdir()
    source = '''from dataclasses import dataclass
@dataclass
class Result:
    axes: dict
    inputs: tuple

def arbitrary_builder(**axes):
    return lambda *inputs: Result(axes, inputs)
'''
    (artifact_dir / 'nested' / 'candidate.py').write_text(source)
    with modules(task, monkeypatch) as contract:
        exporter = importlib.import_module('export_solution')
        wrapper = exporter.tensor_entry({'file': 'nested/candidate.py', 'symbol': 'arbitrary_builder'}, contract.load_workload())
        path = artifact_dir / 'sikl_entry.py'
        path.write_text(wrapper)
        spec = importlib.util.spec_from_file_location('binding_unit_test', path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if name.startswith('gemm'):
            a = types.SimpleNamespace(shape=(2, 4))
            b = types.SimpleNamespace(shape=(3, 4))
            result = mod.run(a, b)
            assert result.axes == {'m': 2, 'n': 3, 'k': 4}
            assert result.inputs == (a, b)
        else:
            a = types.SimpleNamespace(shape=(2, 4))
            w1 = types.SimpleNamespace(shape=(5, 12, 2))
            w2 = types.SimpleNamespace(shape=(5, 4, 3))
            ids = types.SimpleNamespace(shape=(2, 3))
            result = mod.run(a, w1, w2, 'weights', ids, w1_scale='scale1', w2_scale='scale2')
            assert result.axes == {'num_tokens': 2, 'model_dim': 4, 'inter_dim': 6, 'num_experts': 5, 'topk': 3}
            assert result.inputs == (a, w1, w2, 'weights', ids, 'scale1', 'scale2', 0, False)
