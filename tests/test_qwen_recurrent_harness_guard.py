"""Keep recurrent state/routing/oracle inputs and host callbacks frozen."""
import ast
import json
import shutil
from pathlib import Path

import pytest

from src.harness_guard import snapshot_workspace_harness, verify_workspace_harness

TASK = Path(__file__).resolve().parents[1] / "tasks/headkernel/qwen3.8-2.4t__fused_recurrent_gated_delta_rule_decode"
SOURCE = "source/fused_recurrent.py"
KERNEL = "fused_recurrent_gated_delta_rule_packed_decode_kernel"


@pytest.fixture
def workspace(tmp_path):
    path = tmp_path / "task"
    shutil.copytree(TASK, path, symlinks=True)
    # A small stand-in tests the hydrated-input boundary without loading tensors.
    (path / "ut/reference_io.pt").write_bytes(b"immutable captured oracle")
    return path


@pytest.mark.parametrize("name", [
    "ut/reference_io.pt", "ut/meta.json", "ut/selection_validation.json",
    "ut/cases.py", "ut/harness_lib.py", "ut/baseline_ref/fused_recurrent.py.orig",
])
def test_captured_state_routing_reference_and_helpers_are_frozen(workspace, name):
    snapshot = snapshot_workspace_harness(workspace)
    path = workspace / name
    path.write_bytes(path.read_bytes() + b"\nreplacement")
    with pytest.raises(RuntimeError, match="Protected test/harness files changed"):
        verify_workspace_harness(snapshot)


@pytest.mark.parametrize("target", [KERNEL, "fused_recurrent_gated_delta_rule_packed_decode"])
def test_only_device_body_is_editable(workspace, target):
    snapshot = snapshot_workspace_harness(workspace)
    path = workspace / SOURCE
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == target)
    node.body.insert(0, ast.parse("candidate_value = 1").body[0])
    path.write_text(ast.unparse(tree) + "\n")
    if target == KERNEL:
        verify_workspace_harness(snapshot)
    else:
        with pytest.raises(RuntimeError, match="Protected test/harness files changed"):
            verify_workspace_harness(snapshot)


def test_runtime_reports_are_generated_outputs(workspace):
    snapshot = snapshot_workspace_harness(workspace)
    for name in ("ut/negative_check.json", "ut/result.json", "ut/reports/current.json"):
        path = workspace / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"current_run": True}) + "\n")
    verify_workspace_harness(snapshot)
