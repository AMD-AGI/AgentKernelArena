"""Normal local imports in the scheduled FlyDSL targets use the same task policy.

These fixtures exercise the actual source checker, HarnessGuard and compile
process. They never import a candidate or claim GPU qualification.
"""
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from src.harness_guard import snapshot_workspace_harness, verify_workspace_harness
from src.task_protocol import parse_command_result
from src.task_spec import load_task_spec

ROOT = Path(__file__).resolve().parents[1]
TASKS = [
    'triton2flydsl/sglang/fused_dual_residual_rmsnorm',
    'triton2flydsl/sglang/gdn_l2norm_fwd',
    'triton2flydsl/sglang/fused_gdn_gating',
    'triton2flydsl/sglang/fused_norm_gate',
    'torch2flydsl/smoothquant_kernel',
    'torch2flydsl/rmsnorm2d_smoothquant_kernel',
    'torch2flydsl/moe_topk_sigmoid_kernel',
    'triton2flydsl/sglang/merge_state',
    'triton2flydsl/sglang/experts_combine',
    'triton2flydsl/aiter/fused_clamp_act_mul',
    'torch2flydsl/moe_topk_softplus_kernel',
    'torch2flydsl/moe_biased_grouped_topk_kernel',
    'torch2flydsl/rope_thd_fwd_kernel',
    'torch2flydsl/per_1x128_fp8_quant_kernel',
    'torch2flydsl/per_token_fp8_quant_kernel',
    'torch2flydsl/rmsnorm2d_dynamicquant_kernel',
    'torch2flydsl/silu_and_mul_quant_kernel',
]


@pytest.fixture(params=TASKS)
def task(request, tmp_path):
    path = tmp_path / 'task'
    shutil.copytree(ROOT / 'tasks' / request.param, path,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    spec = load_task_spec(path / 'config.yaml', task_id=request.param)
    guard = snapshot_workspace_harness(path, task_spec=spec)
    module_spec = importlib.util.spec_from_file_location('_scheduled_runtime', path / 'task_runtime.py')
    runtime = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(runtime)
    entry, = spec.candidate.entrypoints
    source = path / entry.file
    original = source.read_bytes()
    # This is a legal source-bound fixture, not a numerical implementation.
    source.write_text(f'import flydsl\n\ndef {entry.symbol}(*args, **kwargs):\n    import device_pkg.helper\n    return None\n')
    (path / 'device_pkg').mkdir()
    (path / 'device_pkg/__init__.py').write_text('from . import helper\n')
    (path / 'device_pkg/helper.py').write_text('import flydsl\nTILE = 128\n')
    return path, runtime, guard, source, original


def test_legitimate_package_closure_and_actual_guard(task):
    path, runtime, guard, source, _ = task
    (path / 'device_pkg/helper.py').write_text('from .nested import TILE\n')
    (path / 'device_pkg/nested.py').write_text('from . import helper\nimport flydsl\nTILE = 128\n')
    files = runtime.check_dependencies(runtime.candidate_files(), True)
    assert set(files) == {source, *(path / 'device_pkg' / name for name in ['__init__.py', 'helper.py', 'nested.py'])}
    verify_workspace_harness(guard, discard_added=False)


@pytest.mark.parametrize('forbidden', [
    'import aiter', 'from model import Model', 'from scripts import reference_controls',
    'import importlib as loader\nloader.import_module("aiter")',
    'import torch\ntorch.matmul(a, b)',
])
def test_actual_legal_helper_cannot_hide_forbidden_dependency(task, forbidden):
    path, runtime, guard, _, _ = task
    (path / 'device_pkg/helper.py').write_text('from . import hidden\n')
    (path / 'device_pkg/hidden.py').write_text(forbidden + '\n')
    verify_workspace_harness(guard, discard_added=False)
    with pytest.raises(ValueError, match='Protected dependency|Final operator|Dynamic implementation|Library operator'):
        runtime.check_dependencies(runtime.candidate_files(), True)


@pytest.mark.parametrize('module', ['flydsl', 'torch', 'os'])
def test_local_shadowing_is_not_an_installed_dependency(task, module):
    path, runtime, _, _, _ = task
    (path / 'device_pkg/helper.py').write_text(f'import {module}\n')
    (path / (module + '.py')).write_text('value = 1\n')
    with pytest.raises(ValueError, match='shadows library'):
        runtime.check_dependencies(runtime.candidate_files(), True)


@pytest.mark.parametrize('location', ['declared', 'helper'])
def test_source_symlinks_are_checked_before_resolving(task, location):
    path, runtime, _, source, _ = task
    target = source if location == 'declared' else path / 'device_pkg/helper.py'
    peer = target.with_name('real_' + target.name)
    peer.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(peer.name)
    with pytest.raises(ValueError, match='symlink'):
        runtime.check_dependencies(runtime.candidate_files(), True)


@pytest.mark.parametrize('valid', [True, False])
def test_actual_compile_compiles_helper_without_importing_it(task, valid):
    path, _, _, _, _ = task
    (path / 'device_pkg/helper.py').write_text('raise RuntimeError("compile must not import candidate")\n' if valid else 'def invalid(:\n')
    process = subprocess.run([sys.executable, 'scripts/evaluate.py', 'candidate', 'compile'],
                             cwd=path, text=True, capture_output=True, timeout=30)
    result = parse_command_result(process.stdout, role='candidate', action='compile', returncode=process.returncode)
    assert result.passed is valid
    if valid:
        assert 'device_pkg/helper.py' in result.metadata['sources']
        assert 'device_pkg/__init__.py' in result.metadata['sources']
    else:
        assert 'SyntaxError' in result.reason


def test_existing_initial_baseline_compile_path_is_preserved(task):
    path, _, _, source, original = task
    source.write_bytes(original)
    process = subprocess.run([sys.executable, 'scripts/evaluate.py', 'baseline', 'compile'],
                             cwd=path, text=True, capture_output=True, timeout=30)
    result = parse_command_result(process.stdout, role='baseline', action='compile', returncode=process.returncode)
    assert result.passed, process.stdout + process.stderr


def test_source_review_helper_is_protected(task):
    path, _, guard, _, _ = task
    runtime = path / 'task_runtime.py'
    runtime.write_bytes(runtime.read_bytes() + b'\n')
    with pytest.raises(RuntimeError, match='[Hh]arness|protected'):
        verify_workspace_harness(guard, discard_added=False)
