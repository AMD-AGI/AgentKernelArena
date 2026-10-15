"""Bincount's real harness boundary; CPU checks do not qualify GPU execution."""
import ast
import hashlib
from pathlib import Path
import shutil

import pytest

from src.harness_guard import snapshot_workspace_harness, verify_workspace_harness
from src.task_spec import load_task_spec


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / 'tasks/triton2triton/vllm/triton_bincount'


@pytest.fixture
def boundary(tmp_path):
    workspace = tmp_path / 'workspace'
    shutil.copytree(TASK, workspace)
    spec = load_task_spec(workspace / 'config.yaml', task_id='triton2triton/vllm/triton_bincount')
    edit, = spec.candidate.editable
    assert (edit.scope, edit.symbols, edit.allow_new_helpers) == ('symbols', ('_bincount_kernel',), True)
    return snapshot_workspace_harness(workspace, task_spec=spec), workspace / edit.path


def test_kernel_body_and_new_triton_helpers_remain_editable(boundary):
    snapshot, source = boundary
    tree = ast.parse(source.read_text())
    kernel = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == '_bincount_kernel')
    kernel.body.append(ast.Pass())
    tree.body.extend(ast.parse('@triton.jit\ndef count_helper(value):\n    return value\n').body)
    source.write_text(ast.unparse(ast.fix_missing_locations(tree)))
    verify_workspace_harness(snapshot, discard_added=False)


@pytest.mark.parametrize('mutation', [
    'wrapper', 'reset', 'launch_block_size', 'import', 'module_execution',
    'replace_wrapper_binding', 'replace_kernel_binding', 'helper_executable_default',
])
def test_wrapper_reset_dispatch_and_module_execution_are_protected(boundary, mutation):
    snapshot, source = boundary
    tree = ast.parse(source.read_text())
    wrapper = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name == 'bincount')
    if mutation == 'wrapper':
        wrapper.body = [ast.Return(ast.Constant(None))]
    elif mutation == 'reset':
        reset = next(node for node in wrapper.body if isinstance(node, ast.Assign)
                     and isinstance(node.targets[0], ast.Subscript))
        reset.value = ast.Constant(1)
    elif mutation == 'launch_block_size':
        assignment = next(node for node in wrapper.body if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == 'BLOCK_SIZE'
                                  for target in node.targets))
        assignment.value = ast.Constant(512)
    else:
        additions = {
            'import': 'import os\n',
            'module_execution': 'torch.cuda.synchronize()\n',
            'replace_wrapper_binding': 'bincount = lambda *args: None\n',
            'replace_kernel_binding': '_bincount_kernel = lambda *args: None\n',
            'helper_executable_default': 'def count_helper(value=torch.empty(1)):\n    return value\n',
        }
        tree.body.extend(ast.parse(additions[mutation]).body)
    source.write_text(ast.unparse(ast.fix_missing_locations(tree)))
    with pytest.raises(RuntimeError, match='[Hh]arness|protected|policy'):
        verify_workspace_harness(snapshot, discard_added=False)


def test_boundary_repair_preserves_original_kernel_wrapper_and_timer():
    # This repair changes only the candidate edit policy and its documentation.
    for path, expected in {
        'source/triton_bincount.py': 'dd4e25cd593b43c90da6b205237f61d9378354815ecd0b7548cd94812a4576c5',
        'scripts/task_runner.py': 'd0145c458350c2636669a59f0cae532d53f234bb26454f34eca4d55e2b767309',
    }.items():
        assert hashlib.sha256((TASK / path).read_bytes()).hexdigest() == expected
