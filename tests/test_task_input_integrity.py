"""Task data and references must stay fixed for every optimization integration."""

import json
import logging
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from src.harness_guard import snapshot_workspace_harness, verify_workspace_harness


def _task(tmp_path):
    task = tmp_path / 'task'
    task.mkdir()
    (task / 'config.yaml').write_text(yaml.safe_dump({
        'task_type': 'triton2triton',
        'source_file_path': ['kernel.py'],
        'editable_sources': ['helper.py'],
    }))
    (task / 'kernel.py').write_text('def kernel(): return 1\n')
    (task / 'helper.py').write_text('def helper(): return 1\n')
    (task / 'reference.py').write_text('def reference(): return 1\n')
    (task / 'cases.json').write_text('{"ctx_len": 1024}\n')
    workspace = tmp_path / 'workspace'
    shutil.copytree(task, workspace)
    return task, workspace


@pytest.mark.parametrize('name', ['cases.json', 'reference.py'])
@pytest.mark.parametrize('change', ['edit', 'delete', 'rename'])
def test_task_inputs_are_protected_outside_harness_directories(tmp_path, name, change):
    task, workspace = _task(tmp_path)
    snapshot = snapshot_workspace_harness(workspace, task_root=task)
    path = workspace / name
    if change == 'edit':
        path.write_text('changed')
    elif change == 'delete':
        path.unlink()
    else:
        path.rename(workspace / f'renamed_{name}')
    with pytest.raises(RuntimeError, match=name):
        verify_workspace_harness(snapshot)


def test_declared_edits_and_new_preparation_artifacts_remain_allowed(tmp_path):
    task, workspace = _task(tmp_path)
    (workspace / 'baseline_perf.yaml').write_text('baseline output')
    snapshot = snapshot_workspace_harness(workspace, task_root=task)
    for name in ('kernel.py', 'helper.py'):
        (workspace / name).write_text('def optimized(): return 1\n')
    (workspace / 'forge_driver.py').write_text('# prepared driver\n')
    (workspace / 'agent_status.json').write_text('{}')
    (workspace / 'build').mkdir()
    (workspace / 'build/performance_report.json').write_text('{}')
    verify_workspace_harness(snapshot)
    assert (workspace / 'forge_driver.py').exists()
    assert (workspace / 'agent_status.json').exists()


def test_original_input_paths_remain_protected_after_task_directory_changes(tmp_path):
    task, workspace = _task(tmp_path)
    snapshot = snapshot_workspace_harness(workspace, task_root=task)
    (task / 'cases.json').unlink()
    (workspace / 'cases.json').write_text('{"ctx_len": 16}')
    with pytest.raises(RuntimeError, match='cases.json'):
        verify_workspace_harness(snapshot)


def test_paged_attention_case_reduction_is_rejected_with_ids_unchanged(tmp_path):
    task = Path(__file__).resolve().parents[1] / (
        'tasks/image_kernel/mi355x_vllm_triton_paged_attention_2d'
    )
    workspace = tmp_path / 'task'
    shutil.copytree(task, workspace)
    snapshot = snapshot_workspace_harness(workspace, task_root=task)
    path = workspace / 'session_cases.json'
    data = json.loads(path.read_text())
    for case in data['cases']:
        case['params']['ctx_len'] = 16
    path.write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match='session_cases.json'):
        verify_workspace_harness(snapshot)


@pytest.mark.parametrize('agent_name', ['FORGE', 'CODEX', 'CLAUDE_CODE'])
@pytest.mark.parametrize('stage', ['agent', 'evaluation'])
def test_main_rejects_changed_inputs_before_scoring(tmp_path, agent_name, stage):
    from main import run_task
    from src.module_registration import AgentType

    task, workspace = _task(tmp_path)

    def launcher(**kwargs):
        if stage == 'agent':
            (workspace / 'cases.json').write_text('{"ctx_len": 16}')

    def evaluate(*args, **kwargs):
        (workspace / 'cases.json').write_text('{"ctx_len": 16}')
        return {}

    with (
        patch('main.setup_workspace', return_value=workspace),
        patch('main.evaluate_compilation', return_value=(True, None)),
        patch('main.measure_baseline', return_value=[]),
        patch('main.evaluate_kernel', side_effect=evaluate) as evaluator,
    ):
        completed, _ = run_task(
            eval_config={}, agent=getattr(AgentType, agent_name), agent_launcher=launcher,
            task_name='triton2triton/example', task_config_dir=str(task / 'config.yaml'),
            run_directory=tmp_path / 'run', timestamp='test',
            logger=logging.getLogger(__name__), task_index=1, total_tasks=1,
        )
    assert not completed
    assert evaluator.call_count == (1 if stage == 'evaluation' else 0)
    assert not (workspace / 'task_result.yaml').exists()


def test_cached_dependency_tree_is_not_mistaken_for_task_package(tmp_path):
    task, workspace = _task(tmp_path)
    for root in (task, workspace):
        config_path = root / 'config.yaml'
        config = yaml.safe_load(config_path.read_text())
        config.update(task_type='repository', repo_subdir='vendor/library')
        config_path.write_text(yaml.safe_dump(config))
        (root / 'vendor/library').mkdir(parents=True)
        (root / 'vendor/library/dependency.py').write_text('original')
    snapshot = snapshot_workspace_harness(workspace, task_root=task)
    (workspace / 'vendor/library/dependency.py').write_text('optimized')
    verify_workspace_harness(snapshot)
    assert 'cases.json' in snapshot.digests
