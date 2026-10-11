"""RMSNorm rejects host callbacks before importing or compiling candidate code."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/headkernel/qwen3.8-2.4t__gemma_fused_add_rmsnorm"
RECURRENT = ROOT / "tasks/headkernel/qwen3.8-2.4t__fused_recurrent_gated_delta_rule_decode"
SOURCE = "source/minimax_m3_rmsnorm.py"
REFERENCE = "ut/baseline_ref/minimax_m3_rmsnorm.py.orig"
TARGET = "_gemma_fused_add_rmsnorm_kernel"
GUARDS = ("ut/source_guard.py", "ut/source_guard_policy.json", "ut/cpp_body_guard.py")


@pytest.fixture
def workspace(tmp_path):
    # Only static admission inputs are needed; no runtime, GPU, or Triton import.
    for relative in (SOURCE, REFERENCE, "config.yaml", "scripts/task_runner.py", *GUARDS):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((TASK / relative).read_bytes())
    return tmp_path


@pytest.fixture
def guard(monkeypatch, workspace):
    monkeypatch.syspath_prepend(str(workspace / "ut"))
    spec = importlib.util.spec_from_file_location("rmsnorm_source_guard_test", workspace / "ut/source_guard.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rewrite(workspace, change):
    source = workspace / SOURCE
    tree = ast.parse(source.read_text())
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    change(tree, functions)
    source.write_text(ast.unparse(tree) + "\n")


def prepend(workspace, body):
    def change(tree, functions):
        functions[TARGET].body[:0] = ast.parse(body).body
    rewrite(workspace, change)


def test_policy_reuses_reviewed_grammar_and_binds_existing_frozen_source(guard, workspace):
    for name in ("source_guard.py", "cpp_body_guard.py"):
        assert (TASK / "ut" / name).read_bytes() == (RECURRENT / "ut" / name).read_bytes()
    policy = json.loads((workspace / "ut/source_guard_policy.json").read_text())
    config = yaml.safe_load((workspace / "config.yaml").read_text())
    assert policy["sources"] == {
        SOURCE: {"language": "triton", "targets": [TARGET], "reference": REFERENCE}}
    assert config["harness_protection"]["editable_python_function_bodies"] == {SOURCE: [TARGET]}
    assert config["benchmark_quality"]["validator_sha256"] == hashlib.sha256(
        (workspace / "scripts/task_runner.py").read_bytes()).hexdigest()
    assert (workspace / SOURCE).read_bytes() == (workspace / REFERENCE).read_bytes()
    assert guard.validate_sources(workspace, workspace)


@pytest.mark.parametrize("edit", ["numeric_constexpr", "commute_add", "rsqrt", "gpu_control_flow"])
def test_ordinary_gpu_math_and_constexpr_tuning_remain_editable(guard, workspace, edit):
    def change(tree, functions):
        kernel = functions[TARGET]
        if edit == "numeric_constexpr":
            kernel.body[:0] = ast.parse("tuning: tl.constexpr = 2 * 4").body
        elif edit == "gpu_control_flow":
            kernel.body[:0] = ast.parse("for tile in tl.static_range(1):\n    offset = tile * BLOCK_N").body
        else:
            target = "s" if edit == "commute_add" else "rstd"
            assignment = next(node for node in kernel.body if isinstance(node, ast.Assign)
                              and isinstance(node.targets[0], ast.Name) and node.targets[0].id == target)
            assignment.value = ast.parse("r + x" if edit == "commute_add" else "tl.rsqrt(var + eps)", mode="eval").body
    rewrite(workspace, change)
    assert guard.validate_sources(workspace, workspace)
    result = subprocess.run([sys.executable, "scripts/task_runner.py", "compile"],
                            cwd=workspace, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Compilation: PASS" in result.stdout


CALLBACKS = [
    'probe: tl.constexpr = torch.__dict__["__builtins__"]["exec"]("pass")',
    'probe: tl.constexpr = builtins.exec("pass")',
    'probe: tl.constexpr = exec("pass")',
    'probe: tl.constexpr = torch.cuda.Event.elapsed_time',
    'probe: tl.constexpr = globals()',
    'probe: tl.constexpr = tl.__dict__',
    'probe: tl.constexpr = (tl.sum,)[0](1)',
    'torch.cuda.Event.elapsed_time = 0',
    'import os',
]


@pytest.mark.parametrize("body", CALLBACKS)
def test_host_callbacks_and_introspection_reject_without_execution(guard, workspace, body):
    prepend(workspace, body)
    with pytest.raises(ValueError):
        guard.validate_sources(workspace, workspace)


@pytest.mark.parametrize("mode", ["compile", "correctness", "performance"])
@pytest.mark.parametrize("body", CALLBACKS[:3])
def test_each_configured_phase_rejects_constexpr_callback_before_loading_candidate(workspace, mode, body):
    prepend(workspace, body)
    result = subprocess.run([sys.executable, "scripts/task_runner.py", mode],
                            cwd=workspace, capture_output=True, text=True, timeout=20)
    assert result.returncode != 0
    assert "ValueError:" in result.stderr
    assert "source_guard.py" in result.stderr
    assert "ModuleNotFoundError" not in result.stderr
    # Neither runtime helpers nor GPU libraries are present in this fixture.
    # The source rejection must precede any attempt to enter those phases.
    assert not list((workspace / "build").glob("*"))


@pytest.mark.parametrize("edit", ["launcher", "host_helper", "imports", "signature", "decorator", "other_kernel"])
def test_task_guard_also_freezes_host_and_non_target_source(guard, workspace, edit):
    def change(tree, functions):
        if edit == "imports":
            tree.body.insert(0, ast.Import(names=[ast.alias(name="builtins")]))
        elif edit == "signature":
            functions[TARGET].args.args[0].arg = "different_pointer"
        elif edit == "decorator":
            functions[TARGET].decorator_list = []
        else:
            name = {"launcher": "gemma_fused_add_rmsnorm", "host_helper": "_num_warps",
                    "other_kernel": "_gemma_rmsnorm_kernel"}[edit]
            functions[name].body = ast.parse("return 4").body
    rewrite(workspace, change)
    with pytest.raises(ValueError):
        guard.validate_sources(workspace, workspace)


def test_task_guard_rejects_redirected_source(guard, workspace):
    source = workspace / SOURCE
    source.unlink()
    source.symlink_to(workspace / REFERENCE)
    with pytest.raises(ValueError, match="regular and task-contained"):
        guard.validate_sources(workspace, workspace)
