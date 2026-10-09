"""The observed Qwen UT/routing gap must fail closed without freezing GPU work."""
import ast
import json
import shutil
from pathlib import Path

import pytest

from src.harness_guard import (
    describe_workspace_harness,
    snapshot_workspace_harness,
    verify_workspace_harness,
)
from src.perf_helper_materialization import materialize_perf_helpers_in_workspace


TASK = Path(__file__).resolve().parents[1] / "tasks/headkernel/qwen3.8-2.4t__gemma_fused_add_rmsnorm"
SOURCE = "source/minimax_m3_rmsnorm.py"
ALIAS = "ut/kernel_src/minimax_m3_rmsnorm.py"
EDITABLE = {"_gemma_fused_add_rmsnorm_kernel", "_num_warps"}
CRITICAL = [
    "ut/unittest.py", "ut/harness_lib.py", "ut/cases.py", "ut/meta.json",
    "ut/overlay_setup.py", "ut/leg_runner.py", "ut/baseline_overlay/sitecustomize.py",
    "ut/baseline_overlay/_overlay_manifest.json", "ut/baseline_ref/minimax_m3_rmsnorm.py.orig",
    "ut/layout_validation.json", "ut/selection_validation.json", "ut/runtime_refresh.json",
    "ut/attempts/001_unverified_graph_capture/meta.json",
]


@pytest.fixture
def workspace(tmp_path):
    target = tmp_path / "workspace"
    shutil.copytree(TASK, target, symlinks=True)
    materialize_perf_helpers_in_workspace(target)
    return target


@pytest.mark.parametrize("with_task_root", [False, True])
def test_complete_actual_manifest_matches_default_optimizer_and_report(workspace, with_task_root):
    snapshot = snapshot_workspace_harness(workspace, **({"task_root": TASK} if with_task_root else {}))
    expected = {p.relative_to(workspace).as_posix() for p in workspace.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    expected -= {ALIAS, "ut/negative_check.json"}
    description = describe_workspace_harness(workspace)
    assert set(snapshot.digests) == set(description["protected_paths"]) == expected
    assert set(CRITICAL) <= expected
    assert description["editable_python_function_bodies"] == {SOURCE: sorted(EDITABLE)}
    assert snapshot.source_aliases == {ALIAS: ("../../" + SOURCE, SOURCE)}
    assert snapshot.symlink_protected_sources == (SOURCE,)


@pytest.mark.parametrize("relative", CRITICAL)
@pytest.mark.parametrize("operation", ["change", "delete"])
def test_oracle_routing_cases_baseline_and_evidence_mutations_reject(workspace, relative, operation):
    snapshot = snapshot_workspace_harness(workspace)
    path = workspace / relative
    if operation == "change":
        path.write_bytes(path.read_bytes() + b"\nagent replacement\n")
    else:
        path.unlink()
    with pytest.raises(RuntimeError, match="Protected test/harness files changed"):
        verify_workspace_harness(snapshot)


def rewrite(source, change):
    tree = ast.parse(source.read_text())
    change(tree)
    source.write_text(ast.unparse(tree) + "\n")


@pytest.mark.parametrize("name", sorted(EDITABLE))
def test_gpu_body_and_warp_tuning_remain_editable_through_shipped_alias(workspace, name):
    snapshot = snapshot_workspace_harness(workspace)
    def change(tree):
        node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
        node.body.insert(0, ast.parse("tuning_value = 1").body[0])
    rewrite(workspace / ALIAS, change)
    verify_workspace_harness(snapshot)
    assert (workspace / ALIAS).is_symlink()
    assert (workspace / ALIAS).resolve() == workspace / SOURCE


@pytest.mark.parametrize("attack", ["launcher", "signature", "decorator", "imports", "module_code", "other_kernel"])
def test_source_launcher_interface_and_non_target_code_are_frozen(workspace, attack):
    snapshot = snapshot_workspace_harness(workspace)
    def change(tree):
        functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        kernel = functions["_gemma_fused_add_rmsnorm_kernel"]
        if attack == "launcher":
            functions["gemma_fused_add_rmsnorm"].body = [ast.parse("return x, residual").body[0]]
        elif attack == "signature":
            kernel.args.args[0].arg = "different_interface"
        elif attack == "decorator":
            kernel.decorator_list = []
        elif attack == "imports":
            tree.body.insert(0, ast.Import(names=[ast.alias(name="replacement_oracle")]))
        elif attack == "module_code":
            tree.body.append(ast.parse("gemma_fused_add_rmsnorm = lambda *args: None").body[0])
        else:
            functions["_gemma_rmsnorm_kernel"].body = [ast.Pass()]
    rewrite(workspace / SOURCE, change)
    with pytest.raises(RuntimeError, match="Protected test/harness files changed"):
        verify_workspace_harness(snapshot)


@pytest.mark.parametrize("attack", ["alias_replace", "alias_redirect", "source_redirect"])
def test_candidate_source_alias_and_regular_target_cannot_be_redirected(workspace, attack):
    snapshot = snapshot_workspace_harness(workspace)
    alias, source = workspace / ALIAS, workspace / SOURCE
    if attack == "alias_replace":
        content = alias.read_bytes()
        alias.unlink()
        alias.write_bytes(content)
    elif attack == "alias_redirect":
        alias.unlink()
        alias.symlink_to("../../ut/baseline_ref/minimax_m3_rmsnorm.py.orig")
    else:
        replacement = workspace / "replacement.py"
        source.rename(replacement)
        source.symlink_to("../replacement.py")
    with pytest.raises(RuntimeError, match="source alias|source symlinks"):
        verify_workspace_harness(snapshot)


@pytest.mark.parametrize("with_task_root", [False, True])
def test_actual_runtime_outputs_do_not_change_the_input_contract(workspace, with_task_root):
    snapshot = snapshot_workspace_harness(workspace, **({"task_root": TASK} if with_task_root else {}))
    for name in ("ut/negative_check.json", "ut/result.json", "ut/reports/ledger/hk11.json",
                 "ut/_cand_overlay/sitecustomize.py", "ut/_cand_overlay/_overlay_manifest.json",
                 "build/performance_report.json", ".validator_audit/harness_inventory.json"):
        path = workspace / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"runtime_output": True}) + "\n")
    verify_workspace_harness(snapshot)
    assert set(describe_workspace_harness(workspace)["protected_paths"]) == set(snapshot.digests)


def test_new_ut_helper_is_discarded_before_scoring(workspace):
    snapshot = snapshot_workspace_harness(workspace)
    injected = workspace / "ut/injected_oracle.py"
    injected.write_text("def check(*args): return True\n")
    verify_workspace_harness(snapshot)
    assert not injected.exists()
