"""CPU fixture runners test lifecycle control, not GPU numerical performance."""
import json
from pathlib import Path
import sys

import pytest

from src.task_execution import TaskExecutionError
from src.task_session import TaskSession
from src.task_spec import TaskConfigError, TaskSpec


RUNNER = '''import ast, json, pathlib, sys
args = sys.argv[1:]
role, action = ("task", "validate-task") if args == ["validate-task"] else args
source = pathlib.Path("kernel.py").read_text()
state = "unimplemented" if "NotImplementedError" in source else "implemented"
case = {"test_case_id": "one", "shape": [2], "dtype": "float32", "status": "PASS"}
report = {"protocol": "arena-eval-v1", "role": role, "action": action, "status": "PASS", "cases": [case]}
if role == "task":
    case["checks"] = ["correctness", "performance"]
    report["metadata"] = {"candidate_state": state}
elif action == "compile":
    ast.parse(source)
    report["cases"] = []
else:
    if role == "baseline" and pathlib.Path("provided.txt").exists():
        value = int(pathlib.Path("provided.txt").read_text())
    else:
        scope = {}
        exec(source, scope)
        value = scope["compute"]()
    if action == "correctness" and value != 3:
        case.update(status="FAIL", failure_kind="numerical_mismatch")
        report.update(status="FAIL", reason="Wrong value", failure_kind="numerical_mismatch")
    if action == "performance":
        case.update(execution_time_ms=0.1, benchmark_method="cuda_graph")
print("ARENA_EVAL_RESULT=" + json.dumps(report))
sys.exit(0 if report["status"] == "PASS" else 1)
'''


def create(tmp_path, *, empty=False, provided=None, diagnostic=False):
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    (workspace / "evaluate.py").write_text(RUNNER)
    (workspace / "kernel.py").write_text("def compute():\n    " + ("raise NotImplementedError\n" if empty else "return 3\n"))
    config = {"schema_version": 2,
              "candidate": {"language": "hip", "initial_state": "unimplemented" if empty else "implemented",
                            "editable": ["kernel.py"], "entrypoints": [{"file": "kernel.py", "kind": "function", "symbol": "compute"}]},
              "evaluation": {"runner": [sys.executable, "evaluate.py"]}}
    if provided is not None:
        (workspace / "provided.txt").write_text(str(provided))
        config["baseline"] = {"kind": "provided"}
    if diagnostic:
        config.setdefault("baseline", {}).update(correctness_policy="diagnostic", diagnostic_reason="Unit fixture deviation")
    spec = TaskSpec.from_mapping(config, task_id="suite/cpu_fixture")
    return TaskSession.create(spec, workspace, tmp_path / "framework-state")


def test_baseline_is_a_separate_copy_and_context_contains_complete_manifest(tmp_path):
    session = create(tmp_path)
    initial = session.validate_initial()
    assert initial.accepted
    assert initial.candidate_checks == "verified_as_frozen_baseline"
    (session.workspace / "kernel.py").write_text("def compute(): return 5\n")
    assert "return 3" in (session.baseline_workspace / "kernel.py").read_text()
    session.verify_baseline_sources()
    context = json.loads(session.agent_context_path.read_text())
    assert context["task_id"] == "suite/cpu_fixture"
    assert context["task_config"]["schema_version"] == 2
    assert context["manifest"]["cases"][0]["checks"] == ["correctness", "performance"]
    assert Path(context["baseline_workspace"]) != Path(context["workspace"])


def test_empty_initial_candidate_does_not_skip_final_candidate_checks(tmp_path):
    session = create(tmp_path, empty=True, provided=3)
    initial = session.validate_initial()
    assert initial.accepted
    assert initial.candidate_checks == "candidate_unimplemented"
    assert not any(key[1] == "candidate" for key in session.results)
    with pytest.raises(TaskExecutionError, match="Unimplemented candidate"):
        session.candidate_action("compile")
    (session.workspace / "kernel.py").write_text("def compute(): return 3\n")
    for action in ("compile", "correctness", "performance"):
        assert session.candidate_action(action).result.passed


def test_final_candidate_cannot_inherit_the_baseline_diagnostic_exception(tmp_path):
    session = create(tmp_path, empty=True, provided=4, diagnostic=True)
    initial = session.validate_initial()
    assert initial.accepted
    assert initial.baseline_diagnostic
    assert initial.baseline_numerical_status == "FAIL"
    (session.workspace / "kernel.py").write_text("def compute(): return 4\n")
    assert session.candidate_action("compile").result.passed
    assert not session.candidate_action("correctness").result.passed
    with pytest.raises(TaskExecutionError, match="requires successful correctness"):
        session.candidate_action("performance")


def test_required_baseline_failure_stops_initial_validation(tmp_path):
    session = create(tmp_path, empty=True, provided=4)
    initial = session.validate_initial()
    assert not initial.accepted
    assert ("task_validation", "baseline", "performance") not in session.results
    with pytest.raises(TaskExecutionError, match="accepted initial"):
        session.candidate_action("compile")


def test_declaration_alone_cannot_turn_an_existing_candidate_into_a_stub(tmp_path):
    session = create(tmp_path, empty=True, provided=3)
    (session.baseline_workspace / "kernel.py").write_text("def compute(): return 3\n")
    # Even before interpreting the task's verdict, the original snapshot must
    # still be the one captured at session creation.
    initial = session.validate_initial()
    assert not initial.accepted
    assert "Frozen baseline source changed" in initial.errors[0]


def test_task_state_must_match_the_actual_runner_evidence(tmp_path):
    session = create(tmp_path, empty=True, provided=3)
    config = session.spec.to_mapping()
    config["candidate"]["initial_state"] = "implemented"
    session.spec = TaskSpec.from_mapping(config, task_id=session.spec.task_id)
    initial = session.validate_initial()
    assert not initial.accepted
    assert "verify candidate_state" in initial.errors[0]


def test_changes_between_correctness_and_timing_require_revalidation(tmp_path):
    session = create(tmp_path)
    assert session.validate_initial().accepted
    session.candidate_action("compile")
    session.candidate_action("correctness")
    (session.workspace / "kernel.py").write_text("def compute(): return 100\n")
    with pytest.raises(TaskExecutionError, match="changed after compilation"):
        session.candidate_action("performance")
    session.candidate_action("compile")
    with pytest.raises(TaskExecutionError, match="requires successful correctness"):
        session.candidate_action("performance")


def test_session_refuses_reset_and_nested_snapshot_locations(tmp_path):
    session = create(tmp_path)
    with pytest.raises(FileExistsError):
        TaskSession.create(session.spec, session.workspace, session.state_directory)
    with pytest.raises(TaskConfigError, match="separate directories"):
        TaskSession.create(session.spec, session.workspace, session.workspace / "state")


def test_internal_absolute_links_are_rebound_to_the_frozen_root(tmp_path):
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    (workspace / "source.py").write_text("def compute(): return 3\n")
    (workspace / "kernel.py").symlink_to(workspace / "source.py")
    spec = TaskSpec.from_mapping({"schema_version": 2, "candidate": {"language": "hip", "editable": ["kernel.py"]},
                                 "evaluation": {"runner": ["unused"]}}, task_id="suite/links")
    session = TaskSession.create(spec, workspace, tmp_path / "state")
    (workspace / "source.py").write_text("def compute(): return 9\n")
    assert "return 3" in (session.baseline_workspace / "kernel.py").read_text()
    assert (session.baseline_workspace / "kernel.py").resolve() == session.baseline_workspace / "source.py"


def test_resume_preserves_baseline_and_requires_fresh_candidate_checks(tmp_path):
    session = create(tmp_path)
    assert session.validate_initial().accepted
    session.candidate_action("compile")
    session.candidate_action("correctness")
    (session.workspace / "kernel.py").write_text("def compute(): return 9\n")
    resumed = TaskSession.load(session.spec, session.workspace, session.state_directory)
    assert resumed.initial_validation.accepted
    assert "return 3" in (resumed.baseline_workspace / "kernel.py").read_text()
    with pytest.raises(TaskExecutionError, match="recompile"):
        resumed.candidate_action("performance")
    assert resumed.candidate_action("compile").result.passed
    assert not resumed.candidate_action("correctness").result.passed


def test_resume_rejects_stale_config_and_forged_lifecycle_pass(tmp_path):
    session = create(tmp_path)
    assert session.validate_initial().accepted
    other = session.spec.to_mapping()
    other["candidate"]["language"] = "other_backend"
    with pytest.raises(TaskExecutionError, match="configuration"):
        TaskSession.load(TaskSpec.from_mapping(other, task_id=session.spec.task_id), session.workspace, session.state_directory)
    path = session.state_directory / "initial_validation.json"
    raw = json.loads(path.read_text())
    raw["baseline_numerical_status"] = "FAIL"
    path.write_text(json.dumps(raw))
    with pytest.raises(TaskExecutionError, match="verdict contradicts"):
        TaskSession.load(session.spec, session.workspace, session.state_directory)


def test_resume_rejects_result_tampering_even_when_lifecycle_report_says_pass(tmp_path):
    session = create(tmp_path)
    assert session.validate_initial().accepted
    path = next(session.state_directory.glob("action-*-baseline-performance.json"))
    raw = json.loads(path.read_text())
    raw["result"]["cases"][0]["execution_time_ms"] = 10000
    path.write_text(json.dumps(raw))
    with pytest.raises(TaskExecutionError, match="contradicts its command"):
        TaskSession.load(session.spec, session.workspace, session.state_directory)


def test_failed_initial_validation_still_produces_validator_context(tmp_path):
    session = create(tmp_path, empty=True, provided=4)
    assert not session.validate_initial().accepted
    context = json.loads((session.state_directory / "validation_context.json").read_text())
    assert context["version"] == 1
    assert context["initial_validation"]["baseline_numerical_status"] == "FAIL"
    assert context["actions"][-1]["result"]["status"] == "FAIL"


def test_resume_cannot_recapture_modified_harness_as_original(tmp_path):
    session = create(tmp_path)
    assert session.validate_initial().accepted
    harness = session.workspace / "evaluate.py"
    harness.write_text(harness.read_text() + "\n# changed protected check\n")
    with pytest.raises(RuntimeError, match="Protected test/harness"):
        TaskSession.load(session.spec, session.workspace, session.state_directory)
