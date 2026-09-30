"""CPU packaging checks for functional task drafts; these do not qualify kernels."""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from src.perf_helper_materialization import (
    canonical_aka_helper,
    materialize_perf_helpers_in_workspace,
)
from src.task_protocol import CaseManifest, parse_command_result
from src.task_spec import load_task_spec, resolve_task_path

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / 'tasks/Aiter-task'
TASKS = sorted(p.parents[1] for p in SUITE.glob('*/scripts/workload.json'))


def test_each_task_has_exactly_one_supported_workload_layout():
    assert TASKS
    for config in SUITE.glob('*/config.yaml'):
        root = config.parent
        assert sum((root / path).is_file() for path in
                   ('workload.json', 'scripts/workload.json')) == 1, root.name


@pytest.mark.parametrize('task', TASKS, ids=lambda t: t.name)
def test_config_paths_and_complete_manifest(task):
    spec = load_task_spec(task / 'config.yaml', task_id=f'Aiter-task/{task.name}')
    config = spec.to_mapping()
    assert spec.candidate.initial_state == 'implemented'
    assert spec.candidate.initial_language == 'python'
    assert spec.candidate.language == 'triton'
    assert spec.baseline.kind == 'provided'
    assert spec.baseline.correctness_policy == 'required'
    assert config['platform_support']['required_arch'] == 'gfx950'
    assert {(a.role, a.action) for a in spec.actions} == {
        ('task', 'validate-task'),
        *((role, action) for role in ('baseline', 'candidate')
          for action in ('compile', 'correctness', 'performance')),
    }
    for path in (config['instructions'] + config['baseline']['source_files']
                 + config['candidate']['editable']
                 + [config['evaluation']['workloads'], config['evaluation']['runner'][1]]):
        assert resolve_task_path(task, path, must_exist=True).is_file()
    data = json.loads((task / config['evaluation']['workloads']).read_text())
    assert data['definition']['name'] == task.name
    assert len(data['rows']) == len(data['cases']) == 13
    assert [row['workload']['uuid'] for row in data['rows']] == [
        case['test_case_id'] for case in data['cases']]
    result = parse_command_result('ARENA_EVAL_RESULT=' + json.dumps({
        'protocol': 'arena-eval-v1', 'role': 'task', 'action': 'validate-task',
        'status': 'PASS', 'cases': data['cases'],
    }), role='task', action='validate-task', returncode=0)
    manifest = CaseManifest.from_result(result)
    assert len(manifest.cases) == 13
    assert all(set(case['checks']) == {'correctness', 'performance'}
               for case in manifest.cases)


@pytest.mark.parametrize('task', TASKS, ids=lambda t: t.name)
def test_materialized_copy_is_self_contained(task, tmp_path):
    copy = tmp_path / 'isolated'
    shutil.copytree(task, copy)
    assert not (copy / 'scripts/_aka_benchmark.py').exists()
    changed = materialize_perf_helpers_in_workspace(copy)
    helper = copy / 'scripts/_aka_benchmark.py'
    assert helper in changed
    assert helper.read_text() == canonical_aka_helper(ROOT)
    assert materialize_perf_helpers_in_workspace(copy) == []
    for path in copy.rglob('*.py'):
        tree = ast.parse(path.read_text(), filename=str(path))
        compile(tree, str(path), 'exec')
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert (node.module or '').split('.')[0] not in {'src', 'agents'}
            elif isinstance(node, ast.Import):
                assert not {'src', 'agents'} & {n.name.split('.')[0] for n in node.names}


@pytest.mark.parametrize('task', TASKS, ids=lambda t: t.name)
def test_isolated_cli_rejects_missing_gpu_with_all_case_evidence(task, tmp_path):
    pytest.importorskip('torch')
    copy = tmp_path / 'isolated'
    shutil.copytree(task, copy)
    materialize_perf_helpers_in_workspace(copy)
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    result = subprocess.run([
        sys.executable, '-c',
        "import runpy, sys, torch; torch.cuda.is_available = lambda: False; "
        "sys.argv = ['scripts/task_runner.py', 'validate-task']; "
        "runpy.run_path('scripts/task_runner.py', run_name='__main__')",
    ], cwd=copy, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 1, result.stderr
    report = parse_command_result(result.stdout, role='task', action='validate-task',
                                  returncode=result.returncode)
    assert not report.passed
    assert 'compatible ROCm GPU' in report.reason
    data = json.loads((copy / 'scripts/workload.json').read_text())
    assert [case['test_case_id'] for case in report.cases] == [
        case['test_case_id'] for case in data['cases']]
    assert all(case['status'] == 'FAIL' for case in report.cases)
