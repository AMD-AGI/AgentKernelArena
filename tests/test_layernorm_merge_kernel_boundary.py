"""Kernel edit boundaries through the real guard; CPU checks are not GPU passes."""
import ast
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from src.harness_guard import snapshot_workspace_harness, verify_workspace_harness
from src.task_spec import load_task_spec


ROOT = Path(__file__).resolve().parents[1]
TASKS = [
    ('triton_layernorm_gated', '_layer_norm_fwd_1pass_kernel', 'layer_norm_fwd'),
    ('triton_merge_16x16_to_64x64', 'merge_16x16_to_64x64_inverse_kernel',
     'merge_16x16_to_64x64'),
]


@pytest.fixture(params=TASKS, ids=lambda row: row[0])
def boundary(request, tmp_path):
    name, kernel, wrapper = request.param
    task = ROOT / 'tasks/triton2triton/vllm' / name
    workspace = tmp_path / 'workspace'
    shutil.copytree(task, workspace, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    spec = load_task_spec(workspace / 'config.yaml', task_id=f'triton2triton/vllm/{name}')
    edit, = spec.candidate.editable
    assert edit.path == f'source/{name}.py'
    assert edit.scope == 'symbols' and edit.symbols == (kernel,)
    assert edit.allow_new_helpers is True
    assert [(entry.file, entry.symbol) for entry in spec.candidate.entrypoints] == [(edit.path, kernel)]
    snapshot = snapshot_workspace_harness(workspace, task_spec=spec)
    assert edit.path in snapshot.digests
    verify_workspace_harness(snapshot, discard_added=False)
    return SimpleNamespace(workspace=workspace, source=workspace / edit.path,
                           snapshot=snapshot, kernel=kernel, wrapper=wrapper)


def write_tree(path, tree):
    path.write_text(ast.unparse(ast.fix_missing_locations(tree)) + '\n')


def test_all_existing_device_kernels_are_editable_and_wrappers_are_not(boundary):
    tree = ast.parse(boundary.source.read_text())
    kernels = set()
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        decorators = [value.func if isinstance(value, ast.Call) else value
                      for value in node.decorator_list]
        if any(ast.unparse(value) == 'triton.jit' for value in decorators):
            kernels.add(node.name)
    assert kernels == {boundary.kernel}
    assert any(isinstance(node, ast.FunctionDef) and node.name == boundary.wrapper
               for node in tree.body)


def test_real_guard_accepts_kernel_using_new_triton_helper(boundary):
    tree = ast.parse(boundary.source.read_text())
    kernel = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == boundary.kernel)

    class UseHelper(ast.NodeTransformer):
        changed = False

        def visit_Call(self, node):
            node = self.generic_visit(node)
            if not self.changed and ast.unparse(node.func) == 'tl.program_id':
                self.changed = True
                return ast.Call(func=ast.Name(id='_candidate_identity', ctx=ast.Load()),
                                args=[node], keywords=[])
            return node

    rewrite = UseHelper()
    rewrite.visit(kernel)
    assert rewrite.changed
    tree.body.extend(ast.parse('@triton.jit\ndef _candidate_identity(value):\n    return value\n').body)
    write_tree(boundary.source, tree)
    verify_workspace_harness(boundary.snapshot, discard_added=False)


@pytest.mark.parametrize('mutation', [
    'wrapper_return', 'wrapper_launch', 'import', 'module_execution',
    'replace_wrapper_binding', 'replace_kernel_binding', 'executable_helper_default',
    'noncompiler_decorator',
])
def test_real_guard_rejects_host_wrapper_module_and_definition_policy_changes(boundary, mutation):
    tree = ast.parse(boundary.source.read_text())
    wrapper = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name == boundary.wrapper)
    if mutation == 'wrapper_return':
        wrapper.body = [ast.Return(value=ast.Constant(None))]
    elif mutation == 'wrapper_launch':
        launch = next(node for node in ast.walk(wrapper) if isinstance(node, ast.Subscript)
                      and isinstance(node.value, ast.Name) and node.value.id == boundary.kernel)
        launch.slice = ast.Tuple(elts=[ast.Constant(1)], ctx=ast.Load())
    elif mutation == 'noncompiler_decorator':
        kernel = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                      and node.name == boundary.kernel)
        tree.body.extend(ast.parse('def _candidate_decorator(fn):\n    return fn\n').body)
        kernel.decorator_list.insert(0, ast.Name(id='_candidate_decorator', ctx=ast.Load()))
    else:
        additions = {
            'import': 'import os\n',
            'module_execution': 'torch.cuda.synchronize()\n',
            'replace_wrapper_binding': f'{boundary.wrapper} = lambda *args: None\n',
            'replace_kernel_binding': f'{boundary.kernel} = lambda *args: None\n',
            'executable_helper_default': 'def _candidate_helper(value=torch.empty(1)):\n    return value\n',
        }
        tree.body.extend(ast.parse(additions[mutation]).body)
    write_tree(boundary.source, tree)
    with pytest.raises(RuntimeError, match='[Hh]arness|protected|policy'):
        verify_workspace_harness(boundary.snapshot, discard_added=False)


@pytest.mark.parametrize('relative', [
    'config.yaml', 'README.md', 'workloads.json', '_arena_eval.py',
    '_arena_checks.py', 'scripts/task_runner.py',
])
def test_real_guard_rejects_shipped_contract_reference_and_runner_changes(boundary, relative):
    path = boundary.workspace / relative
    assert relative in boundary.snapshot.digests
    path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(RuntimeError, match='[Hh]arness|protected|policy'):
        verify_workspace_harness(boundary.snapshot, discard_added=False)
