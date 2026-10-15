"""RMSNorm's kernel edit boundary; CPU source checks do not qualify GPU work."""
import ast
from pathlib import Path
import shutil

import pytest

from src.harness_guard import snapshot_workspace_harness, verify_workspace_harness
from src.task_spec import load_task_spec


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / 'tasks/triton2triton/vllm/triton_rms_norm'


@pytest.fixture
def boundary(tmp_path):
    workspace = tmp_path / 'workspace'
    shutil.copytree(TASK, workspace)
    spec = load_task_spec(workspace / 'config.yaml', task_id='triton2triton/vllm/triton_rms_norm')
    edit, = spec.candidate.editable
    assert edit.scope == 'symbols'
    assert edit.symbols == ('_rms_norm_kernel',)
    assert edit.allow_new_helpers is True
    snapshot = snapshot_workspace_harness(workspace, task_spec=spec)
    source = workspace / edit.path
    return snapshot, source


def test_kernel_body_and_new_triton_helpers_are_editable(boundary):
    snapshot, source = boundary
    tree = ast.parse(source.read_text())
    kernel = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == '_rms_norm_kernel')
    kernel.body.append(ast.Pass())
    tree.body.extend(ast.parse('@triton.jit\ndef reduction_helper(value):\n    return value\n').body)
    source.write_text(ast.unparse(ast.fix_missing_locations(tree)))
    verify_workspace_harness(snapshot, discard_added=False)


@pytest.mark.parametrize('mutation', [
    'wrapper', 'launch_block_size', 'import', 'module_execution',
    'replace_wrapper_binding', 'replace_kernel_binding', 'helper_executable_default',
])
def test_wrapper_imports_and_module_execution_are_protected(boundary, mutation):
    snapshot, source = boundary
    tree = ast.parse(source.read_text())
    wrapper = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name == 'rms_norm')
    if mutation == 'wrapper':
        wrapper.body = [ast.Return(ast.Constant(None))]
    elif mutation == 'launch_block_size':
        assignment = next(node for node in wrapper.body if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == 'BLOCK_SIZE'
                                  for target in node.targets))
        assignment.value = ast.Constant(512)
    else:
        additions = {
            'import': 'import os\n',
            'module_execution': 'torch.cuda.synchronize()\n',
            'replace_wrapper_binding': 'rms_norm = lambda *args: None\n',
            'replace_kernel_binding': '_rms_norm_kernel = lambda *args: None\n',
            'helper_executable_default': 'def reduction_helper(value=torch.empty(1)):\n    return value\n',
        }
        tree.body.extend(ast.parse(additions[mutation]).body)
    source.write_text(ast.unparse(ast.fix_missing_locations(tree)))
    with pytest.raises(RuntimeError, match='[Hh]arness|protected|policy'):
        verify_workspace_harness(snapshot, discard_added=False)
