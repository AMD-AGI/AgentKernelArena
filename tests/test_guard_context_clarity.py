"""Guard metadata must describe actual enforcement, not grant edit permissions."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys

import pytest
import yaml

from agents.task_validator.validation_prompt_v2 import build_v2_validation_prompt
from agents.task_validator.trusted_evidence import snapshot_task_evidence
from src.harness_guard import (
    describe_workspace_harness, snapshot_workspace_harness, verify_workspace_harness,
)
from src.perf_helper_materialization import materialize_perf_helpers_in_workspace
from src.task_session import TaskSession
from src.task_spec import TaskSpec, load_task_spec
from test_task_session_v2 import RUNNER
from test_task_validator_v2 import context, draft, normalized

ROOT = Path(__file__).resolve().parents[1]
AST_MODE = "sha256_python_ast_excluding_editable_symbols"


def make_session(tmp_path, *, helpers=False):
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    (workspace / "kernel.py").write_text(
        "import math\nTOLERANCE = 0.01\ndef compute(): return 3\n"
        "def reference(): return 3\n"
    )
    (workspace / "evaluate.py").write_text(RUNNER)
    config = {
        "schema_version": 2,
        "candidate": {"language": "python", "editable": [{
            "path": "kernel.py", "scope": "symbols", "symbols": ["compute"],
            "allow_new_helpers": helpers,
        }]},
        "evaluation": {"runner": [sys.executable, "evaluate.py"]},
    }
    (workspace / "config.yaml").write_text(yaml.safe_dump(config))
    spec = TaskSpec.from_mapping(config, task_id="suite/guard_context")
    return TaskSession.create(spec, workspace, tmp_path / "state")


def add_pass(text, symbol):
    function = next(node for node in ast.parse(text).body
                    if isinstance(node, ast.FunctionDef) and node.name == symbol)
    first = function.body[0]
    lines = text.splitlines(keepends=True)
    lines.insert(first.lineno - 1, " " * first.col_offset + "pass\n")
    return "".join(lines)


@pytest.mark.parametrize("task", [
    "instruction2triton/rocmbench/moe_gemm",
    "triton2triton/rocmbench/hard/moe_gemm",
])
def test_actual_moe_digest_allows_kernel_but_rejects_protected_test(tmp_path, task):
    # Copies of the actual packages, with the canonical helper materializer.
    # No import of Triton, numerical execution, or claim of GPU qualification.
    workspace = tmp_path / "task"
    shutil.copytree(ROOT / "tasks" / task, workspace)
    materialize_perf_helpers_in_workspace(workspace)
    spec = load_task_spec(workspace / "config.yaml", task_id=task)
    snapshot = snapshot_workspace_harness(workspace, task_spec=spec)
    info = describe_workspace_harness(workspace, snapshot=snapshot)
    policy = info["protected_path_policies"]["moe_gemm.py"]
    source = workspace / "moe_gemm.py"
    original = source.read_text()
    assert policy["digest_mode"] == AST_MODE
    assert policy["digest"] == snapshot.digests["moe_gemm.py"]
    assert policy["digest"] != hashlib.sha256(source.read_bytes()).hexdigest()
    assert policy["editable_symbols"] == ["moe_gemm_kernel"]
    assert policy["allow_new_helpers"] is True
    assert "test_correctness" in policy["initial_top_level_names"]
    source.write_text(add_pass(original, "moe_gemm_kernel"))
    verify_workspace_harness(snapshot)
    assert describe_workspace_harness(workspace, snapshot=snapshot) == info
    source.write_text(add_pass(original, "test_correctness"))
    with pytest.raises(RuntimeError, match="moe_gemm.py"):
        verify_workspace_harness(snapshot)


def test_raw_file_metadata_corresponds_to_byte_protection(tmp_path):
    session = make_session(tmp_path)
    guard = describe_workspace_harness(session.workspace, snapshot=session.harness)
    policy = guard["protected_path_policies"]["evaluate.py"]
    runner = session.workspace / "evaluate.py"
    assert policy == {
        "digest_mode": "sha256_bytes",
        "digest": hashlib.sha256(runner.read_bytes()).hexdigest(),
        "editable_symbols": [], "allow_new_helpers": False, "initial_top_level_names": [],
    }
    runner.write_text(runner.read_text() + "# even a byte-only change is protected\n")
    with pytest.raises(RuntimeError, match="evaluate.py"):
        session.verify_candidate_harness()


@pytest.mark.parametrize("helpers", [False, True])
@pytest.mark.parametrize("new_node", [
    "def helper(): return 3\n", "async def helper(): return 3\n", "class Helper: pass\n",
])
def test_helper_policy_matches_actual_new_node_enforcement(tmp_path, helpers, new_node):
    session = make_session(tmp_path, helpers=helpers)
    info = describe_workspace_harness(session.workspace, snapshot=session.harness)
    policy = info["protected_path_policies"]["kernel.py"]
    assert policy["allow_new_helpers"] is helpers
    assert policy["initial_top_level_names"] == ["TOLERANCE", "compute", "math", "reference"]
    source = session.workspace / "kernel.py"
    source.write_text(source.read_text() + new_node)
    if helpers:
        session.verify_candidate_harness()
    else:
        with pytest.raises(RuntimeError, match="kernel.py"):
            session.verify_candidate_harness()
    # New helper names must not silently enter the original-name inventory.
    assert describe_workspace_harness(session.workspace, snapshot=session.harness) == info


@pytest.mark.parametrize("old,new", [
    ("def reference(): return 3", "def reference(): return 9"),
    ("import math", "import random"),
    ("TOLERANCE = 0.01", "TOLERANCE = 1"),
])
def test_allowing_new_helpers_keeps_original_helpers_imports_constants_protected(tmp_path, old, new):
    session = make_session(tmp_path, helpers=True)
    source = session.workspace / "kernel.py"
    source.write_text(source.read_text().replace(old, new))
    with pytest.raises(RuntimeError, match="kernel.py"):
        session.verify_candidate_harness()


@pytest.mark.parametrize("receipt_mode", ["current", "older_without_description", "forged_description"])
def test_session_receipt_context_and_resume_use_original_snapshot(tmp_path, receipt_mode):
    session = make_session(tmp_path, helpers=True)
    assert session.validate_initial().accepted
    initial = session.validation_context()["harness"]
    receipt_path = session.state_directory / "harness.json"
    receipt = json.loads(receipt_path.read_text())
    assert receipt["effective_guard"] == initial
    assert {p: v["digest"] for p, v in initial["protected_path_policies"].items()} == receipt["digests"]
    assert json.loads((session.state_directory / "validation_context.json").read_text())["harness"] == initial
    if receipt_mode == "older_without_description":
        del receipt["effective_guard"]
    elif receipt_mode == "forged_description":
        receipt["effective_guard"] = {"protected_path_policies": {}, "editable_entrypoint_targets": {}}
    receipt_path.write_text(json.dumps(receipt))
    source = session.workspace / "kernel.py"
    source.write_text(source.read_text().replace("def compute(): return 3", "def compute(): return helper()")
                      + "def helper(): return 3\n")
    resumed = TaskSession.load(session.spec, session.workspace, session.state_directory)
    assert resumed.validation_context()["harness"] == initial
    # A candidate-authored config must not redefine the exported effective guard.
    config_path = session.workspace / "config.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["candidate"]["editable"] = ["kernel.py", "evaluate.py"]
    config_path.write_text(yaml.safe_dump(config))
    assert resumed.validation_context()["harness"] == initial
    with pytest.raises(RuntimeError, match="config.yaml"):
        resumed.verify_candidate_harness()


def test_snapshot_of_a_different_workspace_is_not_mislabelled(tmp_path):
    session = make_session(tmp_path)
    with pytest.raises(ValueError, match="different workspace"):
        describe_workspace_harness(tmp_path, snapshot=session.harness)


def test_file_editable_candidate_is_not_described_as_protected(tmp_path):
    session = make_session(tmp_path)
    config = session.spec.to_mapping()
    config["candidate"]["editable"] = ["kernel.py"]
    (session.workspace / "config.yaml").write_text(yaml.safe_dump(config))
    spec = TaskSpec.from_mapping(config, task_id=session.spec.task_id)
    snapshot = snapshot_workspace_harness(session.workspace, task_spec=spec)
    info = describe_workspace_harness(session.workspace, snapshot=snapshot)
    assert "kernel.py" not in info["protected_path_policies"]
    assert info["editable_entrypoint_targets"] == {}
    (session.workspace / "kernel.py").write_text("# file-scope edit\n")
    verify_workspace_harness(snapshot)


def test_prompt_includes_effective_symbol_modes_beyond_the_path_sample(tmp_path):
    session = make_session(tmp_path)
    assert session.validate_initial().accepted
    ctx = session.validation_context()
    # Large protected-path inventories may omit the candidate from the sample.
    ctx["harness"]["protected_paths"] = [f"a-{i}.py" for i in range(30)] + ["kernel.py"]
    evidence = snapshot_task_evidence(ctx, task_id=session.spec.task_id)
    prompt = build_v2_validation_prompt(
        task_id=session.spec.task_id, task_config=session.spec.to_mapping(),
        workspace=str(session.workspace), trusted_task_evidence=evidence,
    )
    data = prompt.split("FRAMEWORK TRANSPORT AND CAPTURED ACTION SUMMARY (JSON data)\n", 1)[1]
    transport = json.loads(data.split("\n\nTASK DECLARATION", 1)[0])
    assert "kernel.py" not in transport["harness"]["protected_paths_sample"]
    assert transport["harness"]["symbol_digest_policies"]["kernel.py"] == ctx["harness"]["protected_path_policies"]["kernel.py"]
    assert "a single\n   digest does not imply a whole-file lock" in prompt
    assert "missing or contradictory" in prompt


@pytest.mark.parametrize("verdict", ["FAIL", "WARN"])
def test_metadata_never_overrides_independent_harness_review(tmp_path, verdict):
    session_path = tmp_path / "guard"; session_path.mkdir()
    session = make_session(session_path)
    ctx = context(tmp_path, state="implemented", baseline_kind="initial_candidate")
    ctx["harness"] = describe_workspace_harness(session.workspace, snapshot=session.harness)
    raw = draft(ctx)
    raw["checks"]["harness_integrity"]["status"] = verdict
    raw["checks"]["harness_integrity"]["editable_targets_preserved"] = verdict != "FAIL"
    report = normalized(ctx, raw)
    assert report["overall_status"] == verdict
