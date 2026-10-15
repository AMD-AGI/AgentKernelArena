"""Exercise the real edit guard without importing or compiling GPU kernels."""
import ast
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from src.harness_guard import snapshot_workspace_harness, verify_workspace_harness
from src.task_spec import load_task_spec


ROOT = Path(__file__).resolve().parents[1]
TASKS = [
    ('triton_ep_scatter_2', '_fwd_kernel_ep_scatter_2', 'ep_scatter_2'),
    ('triton_per_token_group_quant_fp8', '_per_token_group_quant_fp8', 'per_token_group_quant_fp8'),
    ('triton_reduce_segments', 'reduce_segments', 'reduce_attention_segments'),
    ('triton_fla_layernorm_gated', 'layer_norm_gated_fwd_kernel', 'layer_norm_gated_fwd'),
    ("triton_causal_conv1d_fwd", "_causal_conv1d_fwd_kernel", "causal_conv1d_fwd"),
    ("triton_log_softmax", "_log_softmax_kernel", "log_softmax"),
    ("triton_ranks", "_ranks_kernel", "compute_ranks"),
    ("triton_rejection_greedy_sample", "rejection_greedy_sample_kernel",
     "rejection_greedy_sample"),
    ("triton_rejection_random_sample", "rejection_random_sample_kernel",
     "rejection_random_sample"),
    ("triton_reshape_and_cache_flash", "reshape_and_cache_kernel_flash",
     "reshape_and_cache_flash"),
]


@pytest.fixture(params=TASKS, ids=lambda row: row[0])
def boundary(request, tmp_path):
    name, kernel, wrapper = request.param
    task = ROOT / "tasks/triton2triton/vllm" / name
    workspace = tmp_path / "workspace"
    shutil.copytree(task, workspace, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    spec = load_task_spec(workspace / "config.yaml", task_id=f"triton2triton/vllm/{name}")
    edit, = spec.candidate.editable
    assert edit.path == f"source/{name}.py"
    allowed = ("cdiv_fn", "find_seq_idx", "reduce_segments") if name == "triton_reduce_segments" else (kernel,)
    assert edit.scope == "symbols" and edit.symbols == allowed
    assert edit.allow_new_helpers is True
    assert [(entry.file, entry.symbol) for entry in spec.candidate.entrypoints] == [(edit.path, kernel)]
    snapshot = snapshot_workspace_harness(workspace, task_spec=spec)
    assert edit.path in snapshot.digests
    verify_workspace_harness(snapshot, discard_added=False)
    return SimpleNamespace(workspace=workspace, source=workspace / edit.path,
                           snapshot=snapshot, kernel=kernel, wrapper=wrapper, allowed=allowed)


def write_tree(path, tree):
    path.write_text(ast.unparse(ast.fix_missing_locations(tree)) + "\n")


def test_boundary_includes_every_existing_device_kernel_and_keeps_wrapper(boundary):
    tree = ast.parse(boundary.source.read_text())
    kernels = set()
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        decorators = [value.func if isinstance(value, ast.Call) else value
                      for value in node.decorator_list]
        if any(ast.unparse(value) == "triton.jit" for value in decorators):
            kernels.add(node.name)
    assert kernels == set(boundary.allowed)
    assert any(isinstance(node, ast.FunctionDef) and node.name == boundary.wrapper
               for node in tree.body)


def test_guard_accepts_kernel_body_using_new_device_helper(boundary):
    tree = ast.parse(boundary.source.read_text())
    kernel = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == boundary.kernel)

    class UseHelper(ast.NodeTransformer):
        changed = False

        def visit_Call(self, node):
            node = self.generic_visit(node)
            if not self.changed and ast.unparse(node.func) == "tl.program_id":
                self.changed = True
                return ast.Call(func=ast.Name(id="_candidate_identity", ctx=ast.Load()),
                                args=[node], keywords=[])
            return node

    rewrite = UseHelper()
    rewrite.visit(kernel)
    assert rewrite.changed
    tree.body.extend(ast.parse("@triton.jit\ndef _candidate_identity(value):\n    return value\n").body)
    write_tree(boundary.source, tree)
    verify_workspace_harness(boundary.snapshot, discard_added=False)


@pytest.mark.parametrize("mutation", [
    "wrapper_return", "wrapper_launch", "import_oracle", "comparison_hook",
    "nonfinite_hook", "replace_wrapper_binding", "replace_kernel_binding",
    "executable_helper_default", "noncompiler_decorator",
])
def test_guard_rejects_wrapper_or_comparison_policy_changes(boundary, mutation):
    tree = ast.parse(boundary.source.read_text())
    wrapper = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name == boundary.wrapper)
    if mutation == "wrapper_return":
        wrapper.body = ast.parse("return torch.zeros(1)").body
    elif mutation == "wrapper_launch":
        launch = next(node for node in ast.walk(wrapper) if isinstance(node, ast.Subscript)
                      and isinstance(node.value, ast.Name) and node.value.id == boundary.kernel)
        launch.slice = ast.Tuple(elts=[ast.Constant(1)], ctx=ast.Load())
    elif mutation == "noncompiler_decorator":
        kernel = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                      and node.name == boundary.kernel)
        tree.body.extend(ast.parse("def _candidate_decorator(fn):\n    return fn\n").body)
        kernel.decorator_list.insert(0, ast.Name(id="_candidate_decorator", ctx=ast.Load()))
    else:
        additions = {
            "import_oracle": "import _arena_checks\n",
            "comparison_hook": "torch.testing.assert_close = lambda *args, **kwargs: None\n",
            "nonfinite_hook": "torch.isfinite = lambda value: torch.ones_like(value, dtype=torch.bool)\n",
            "replace_wrapper_binding": f"{boundary.wrapper} = lambda *args: None\n",
            "replace_kernel_binding": f"{boundary.kernel} = lambda *args: None\n",
            "executable_helper_default": "def _candidate_helper(value=torch.empty(1)):\n    return value\n",
        }
        tree.body.extend(ast.parse(additions[mutation]).body)
    write_tree(boundary.source, tree)
    with pytest.raises(RuntimeError, match="[Hh]arness|protected|policy"):
        verify_workspace_harness(boundary.snapshot, discard_added=False)


@pytest.mark.parametrize("relative", [
    "config.yaml", "README.md", "workloads.json", "_arena_eval.py",
    "scripts/task_runner.py",
])
def test_guard_rejects_changes_to_contract_or_evaluation_path(boundary, relative):
    path = boundary.workspace / relative
    assert relative in boundary.snapshot.digests
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(RuntimeError, match="[Hh]arness|protected|policy"):
        verify_workspace_harness(boundary.snapshot, discard_added=False)


def test_guard_rejects_reference_change(boundary):
    references = [path for path in ("_arena_checks.py", "_arena_contract.py")
                  if (boundary.workspace / path).is_file()]
    assert references
    path = boundary.workspace / references[0]
    assert references[0] in boundary.snapshot.digests
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(RuntimeError, match="[Hh]arness|protected|policy"):
        verify_workspace_harness(boundary.snapshot, discard_added=False)


@pytest.mark.parametrize("name,constant", [
    ("triton_causal_conv1d_fwd", "PAD_SLOT_ID"),
    ("triton_rejection_greedy_sample", "PLACEHOLDER_TOKEN_ID"),
    ("triton_rejection_random_sample", "PLACEHOLDER_TOKEN_ID"),
])
def test_guard_keeps_module_constants_immutable(tmp_path, name, constant):
    workspace = tmp_path / "workspace"
    shutil.copytree(ROOT / "tasks/triton2triton/vllm" / name, workspace,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    spec = load_task_spec(workspace / "config.yaml", task_id=f"triton2triton/vllm/{name}")
    snapshot = snapshot_workspace_harness(workspace, task_spec=spec)
    source = workspace / f"source/{name}.py"
    tree = ast.parse(source.read_text())
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == constant
                              for target in node.targets))
    assignment.value = ast.Constant(0)
    write_tree(source, tree)
    with pytest.raises(RuntimeError, match="[Hh]arness|protected|policy"):
        verify_workspace_harness(snapshot, discard_added=False)
