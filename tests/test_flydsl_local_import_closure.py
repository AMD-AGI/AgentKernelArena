"""Task-owned import review checks; these CPU fixtures never execute GPU code."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from src.harness_guard import snapshot_workspace_harness, verify_workspace_harness
from src.task_spec import load_task_spec
from src.task_protocol import parse_command_result

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['fused_add_rmsnorm_kernel', 'layernorm2d_kernel', 'layernorm2d_with_add_kernel',
         'rmsnorm2d_kernel', 'per_token_i8_quant_kernel', 'rope_fwd_kernel',
         'silu_and_mul_kernel', 'swiglu_and_mul_kernel']


@pytest.fixture(params=NAMES)
def task(request, tmp_path):
    path = tmp_path / request.param
    shutil.copytree(ROOT / 'tasks/torch2flydsl' / request.param, path,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    module_spec = importlib.util.spec_from_file_location('_local_import_task', path / 'task_runtime.py')
    runtime = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(runtime)
    spec = load_task_spec(path / 'config.yaml', task_id='torch2flydsl/' + request.param)
    guard = snapshot_workspace_harness(path, task_spec=spec)
    entry = spec.candidate.entrypoints[0].symbol
    (path / 'kernel.py').write_text(f'import flydsl\n\ndef {entry}(*args, **kwargs):\n    return None\n')
    return path, runtime, guard


def write(path, relative, content):
    file = path / relative
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(content)


@pytest.mark.parametrize('layout', ['one_hop', 'package_relative', 'namespace', 'cycle', 'constant'])
def test_accepts_legitimate_new_local_device_helpers(task, layout):
    path, runtime, guard = task
    imports = {'one_hop': 'import device_helper', 'package_relative': 'from device_pkg import helper',
               'namespace': 'import device_pkg.helper', 'cycle': 'import helper_a',
               'constant': 'from device_helper import TILE'}
    with (path / 'kernel.py').open('a') as f:
        f.write(imports[layout] + '\n')
    if layout in {'one_hop', 'constant'}:
        write(path, 'device_helper.py', 'import flydsl\nTILE = 128\n')
    elif layout in {'package_relative', 'namespace'}:
        if layout == 'package_relative':
            write(path, 'device_pkg/__init__.py', 'from . import helper\n')
        write(path, 'device_pkg/helper.py', 'from .inner import tile\n')
        write(path, 'device_pkg/inner.py', 'import flydsl\ntile = 128\n')
    else:
        write(path, 'helper_a.py', 'import helper_b\n')
        write(path, 'helper_b.py', 'import helper_a\nimport flydsl\n')
    files = runtime.check_dependencies(runtime.candidate_files(), True)
    assert path / 'kernel.py' in files and len(files) == len(set(files)) > 1
    assert all(file.is_relative_to(path) for file in files)
    verify_workspace_harness(guard, discard_added=False)


@pytest.mark.parametrize('forbidden', [
    'import aiter', 'from model import Model', 'from scripts import reference_controls',
    'import importlib as loader\nloader.import_module("aiter")',
    'import torch\ntorch.matmul(a, b)',
])
@pytest.mark.parametrize('nested', [False, True])
def test_rejects_hidden_forbidden_dependencies_in_normal_import_closure(task, forbidden, nested):
    path, runtime, _ = task
    with (path / 'kernel.py').open('a') as f:
        f.write('from device_pkg import helper\n')
    write(path, 'device_pkg/__init__.py', '')
    write(path, 'device_pkg/helper.py', 'from . import hidden\n' if nested else forbidden + '\n')
    if nested:
        write(path, 'device_pkg/hidden.py', forbidden + '\n')
    with pytest.raises(ValueError, match='Protected dependency|Final operator|Dynamic implementation|Library operator'):
        runtime.check_dependencies(runtime.candidate_files(), True)


@pytest.mark.parametrize('module', ['flydsl', 'torch', 'os'])
def test_rejects_task_local_runtime_library_shadowing(task, module):
    path, runtime, _ = task
    with (path / 'kernel.py').open('a') as f:
        f.write(f'import {module}\n')
    write(path, module + '.py', 'value = 1\n')
    with pytest.raises(ValueError, match='shadows library'):
        runtime.check_dependencies(runtime.candidate_files(), True)


@pytest.mark.parametrize('mode', ['file_symlink', 'directory_symlink', 'relative_escape', 'missing_local'])
def test_rejects_escaped_or_invalid_local_sources(task, tmp_path, mode):
    path, runtime, _ = task
    outside = tmp_path / 'outside'
    outside.mkdir()
    write(outside, 'helper.py', 'import flydsl\n')
    if mode == 'file_symlink':
        (path / 'device_helper.py').symlink_to(outside / 'helper.py')
        statement = 'import device_helper'
    elif mode == 'directory_symlink':
        (path / 'device_pkg').symlink_to(outside)
        statement = 'import device_pkg.helper'
    elif mode == 'relative_escape':
        write(path, 'device_pkg/__init__.py', 'from ... import helper\n')
        statement = 'import device_pkg'
    else:
        write(path, 'device_pkg/__init__.py', '')
        statement = 'import device_pkg.missing'
    with (path / 'kernel.py').open('a') as f:
        f.write(statement + '\n')
    with pytest.raises((ValueError, OSError), match='symlink|escapes|Missing task-local'):
        runtime.check_dependencies(runtime.candidate_files(), True)


@pytest.mark.parametrize('valid', [True, False])
def test_actual_compile_process_checks_new_helper_sources_without_importing_them(task, valid):
    path, _, _ = task
    with (path / 'kernel.py').open('a') as f:
        f.write('import device_helper\n')
    write(path, 'device_helper.py', 'raise RuntimeError("must not import during compile")\n' if valid else 'def broken(:\n')
    process = subprocess.run([sys.executable, 'scripts/evaluate.py', 'candidate', 'compile'],
                             cwd=path, text=True, capture_output=True, timeout=30)
    result = parse_command_result(process.stdout, role='candidate', action='compile',
                                  returncode=process.returncode)
    assert result.passed is valid
    if valid:
        assert set(result.metadata['sources']) == {'kernel.py', 'device_helper.py'}
    else:
        assert 'SyntaxError' in result.reason


def test_added_task_helper_is_protected_after_materialization(task):
    path, _, guard = task
    source = path / 'task_runtime.py'
    source.write_bytes(source.read_bytes() + b'\n')
    with pytest.raises(RuntimeError, match='[Hh]arness|protected'):
        verify_workspace_harness(guard, discard_added=False)


def test_declared_candidate_symlink_is_not_dereferenced_before_source_review(task):
    path, runtime, _ = task
    (path / 'real_candidate.py').write_text('import flydsl\n')
    (path / 'kernel.py').unlink()
    (path / 'kernel.py').symlink_to('real_candidate.py')
    assert runtime.candidate_files() == [path / 'kernel.py']
    with pytest.raises(ValueError, match='symlink'):
        runtime.check_dependencies(runtime.candidate_files(), True)
