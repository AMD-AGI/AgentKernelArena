"""CPU process fixtures for orchestration; timings are synthetic, not GPU evidence."""
import json
import logging
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import yaml

from src.evaluator import evaluate_task_session
from src.task_run import run_task_v2, task_run_is_complete
from tests.test_task_session_v2 import RUNNER, create


@pytest.fixture(autouse=True)
def cpu_runtime(monkeypatch):
    monkeypatch.setattr("src.task_runtime._runtime_identity", lambda: {
        "gpu_arch": "gfx950", "gpu_name": "CPU protocol fixture, no GPU", "torch": "fixture"})
    monkeypatch.setenv("AGENT_KERNEL_ARENA_PYTHON", sys.executable)


def package(tmp_path, *, empty=False, provided=None, diagnostic=False, exporter=None):
    root = tmp_path / "task-package"
    root.mkdir()
    (root / "evaluate.py").write_text(RUNNER)
    (root / "kernel.py").write_text("def compute():\n    " + (
        "raise NotImplementedError\n" if empty else "return 3\n"))
    config = {"schema_version": 2,
              "candidate": {"language": "hip", "initial_state": "unimplemented" if empty else "implemented",
                            "editable": ["kernel.py"],
                            "entrypoints": [{"file": "kernel.py", "kind": "function", "symbol": "compute"}]},
              "evaluation": {"runner": [sys.executable, "evaluate.py"]}}
    if provided is not None:
        (root / "provided.txt").write_text(str(provided))
        config["baseline"] = {"kind": "provided"}
    if diagnostic:
        config["baseline"].update(correctness_policy="diagnostic", diagnostic_reason="CPU fixture mismatch")
    if exporter:
        (root / "export.py").write_text(exporter)
        config["exports"] = [{"format": "fixture", "output": "artifacts/solution.json",
                              "command": [sys.executable, "export.py"]}]
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def run(tmp_path, path, launcher, agent="codex"):
    return run_task_v2(eval_config={"agent": {"template": agent}}, agent=SimpleNamespace(value=agent),
                       agent_launcher=launcher, task_name="suite/protocol_fixture",
                       task_config_dir=str(path), run_directory=tmp_path / "run", timestamp="fixture",
                       logger=logging.getLogger("test_task_run_v2"))


def read_report(workspace):
    return yaml.safe_load((workspace / "task_result.yaml").read_text())


def test_complete_pipeline_keeps_baseline_and_evaluates_after_agent_error(tmp_path):
    path = package(tmp_path)

    def launcher(**kwargs):
        context = json.loads(Path(os.environ["ARENA_TASK_CONTEXT"]).read_text())
        assert context["workspace"] == kwargs["workspace"]
        assert context["baseline_workspace"] != kwargs["workspace"]
        (Path(kwargs["workspace"]) / "kernel.py").write_text("def compute(): return 1 + 2\n")
        raise RuntimeError("Model CLI exited after delivering candidate")

    complete, workspace = run(tmp_path, path, launcher)
    report = read_report(workspace)
    assert complete and report["candidate_accepted"]
    assert report["score"] == 220
    assert report["agent_execution"]["status"] == "FAILED"
    assert report["baseline_correctness"]["status"] == "PASS"
    assert report["delivery_status"] == "COMPLETE"
    assert task_run_is_complete(workspace, "suite/protocol_fixture", "codex")
    assert not task_run_is_complete(workspace, "suite/protocol_fixture", "claude_code")
    candidate = workspace / "kernel.py"
    original = candidate.read_text()
    candidate.write_text("def compute(): return 9\n")
    assert not task_run_is_complete(workspace, "suite/protocol_fixture", "codex")
    candidate.write_text(original)
    report["score"] = 9999
    (workspace / "task_result.yaml").write_text(yaml.safe_dump(report))
    assert not task_run_is_complete(workspace, "suite/protocol_fixture", "codex")


def test_empty_candidate_is_valid_initially_but_not_a_successful_submission(tmp_path):
    path = package(tmp_path, empty=True, provided=3)
    complete, workspace = run(tmp_path, path, lambda **_: None)
    report = read_report(workspace)
    assert complete
    assert report["initial_task_validation"]["accepted"]
    assert not report["pass_compilation"]
    assert not report["candidate_accepted"]
    assert report["score"] == 0


def test_diagnostic_baseline_failure_stays_visible_with_passing_candidate(tmp_path):
    path = package(tmp_path, empty=True, provided=4, diagnostic=True)

    def launcher(**kwargs):
        (Path(kwargs["workspace"]) / "kernel.py").write_text("def compute(): return 3\n")

    _, workspace = run(tmp_path, path, launcher)
    report = read_report(workspace)
    assert report["candidate_accepted"]
    assert report["baseline_correctness"]["status"] == "FAIL"
    assert report["initial_task_validation"]["baseline_diagnostic"]


def test_invalid_initial_task_never_starts_optimization(tmp_path):
    path = package(tmp_path, empty=True, provided=4)

    def launcher(**_):
        pytest.fail("Invalid task must not spend an optimization budget")

    _, workspace = run(tmp_path, path, launcher)
    report = read_report(workspace)
    assert report["agent_execution"]["status"] == "NOT_RUN"
    assert report["framework_error"].startswith("RuntimeError: Initial task validation failed")
    assert not report["candidate_accepted"]


def test_changed_harness_is_rejected_by_official_evaluation(tmp_path):
    path = package(tmp_path)

    def launcher(**kwargs):
        target = Path(kwargs["workspace"]) / "evaluate.py"
        target.write_text(target.read_text() + "\n# unauthorized edit\n")

    _, workspace = run(tmp_path, path, launcher)
    report = read_report(workspace)
    assert "Protected test/harness" in report["framework_error"]
    assert report["score"] == 0


def test_escaping_candidate_retains_a_failed_report_without_evaluating_it(tmp_path):
    path = package(tmp_path)
    outside = tmp_path / "external.py"
    outside.write_text("raise AssertionError('external code must not run')\n")

    def launcher(**kwargs):
        target = Path(kwargs["workspace"]) / "kernel.py"
        target.unlink()
        target.symlink_to(outside)

    complete, workspace = run(tmp_path, path, launcher)
    report = read_report(workspace)
    assert complete and report["score"] == 0
    assert not report["candidate_accepted"]
    assert report["evaluated_candidate_sources"] is None
    assert "within workspace" in report["candidate_source_error"]
    state = workspace.parent / ".task-sessions" / workspace.name
    assert not list(state.glob("action-*-candidate-*.json"))
    assert json.loads((state / "completion.json").read_text())["candidate_source_error"]


def test_resume_retains_original_baseline_and_performs_new_candidate_checks(tmp_path):
    path = package(tmp_path)
    _, workspace = run(tmp_path, path, lambda **_: None)
    before = json.loads((workspace.parent / ".task-sessions" / workspace.name / "initial_sources.json").read_text())

    def launcher(**kwargs):
        (Path(kwargs["workspace"]) / "kernel.py").write_text("def compute(): return 9\n")

    _, workspace = run(tmp_path, path, launcher)
    report = read_report(workspace)
    assert not report["candidate_accepted"]
    assert report["baseline_correctness"]["status"] == "PASS"
    assert report["score"] == 20
    after = json.loads((workspace.parent / ".task-sessions" / workspace.name / "initial_sources.json").read_text())
    assert before == after


@pytest.mark.parametrize("modify_report", [False, "delete", "symlink", "directory"])
def test_common_export_for_any_agent_and_report_tampering_detection(tmp_path, modify_report):
    code = '''import json, os, pathlib
output = pathlib.Path(os.environ["ARENA_EXPORT_PATH"])
output.parent.mkdir(parents=True, exist_ok=True)
report = pathlib.Path(os.environ["ARENA_FINAL_RESULT_PATH"])
output.write_text(json.dumps({"delivered": True}))
'''
    if modify_report:
        code += "report.unlink()\n"
    if modify_report == "symlink":
        code += "report.symlink_to(output)\n"
    elif modify_report == "directory":
        code += "report.mkdir()\n(report / 'diagnostic.txt').write_text('exporter output')\n"
    path = package(tmp_path, exporter=code)
    _, workspace = run(tmp_path, path, lambda **_: None, agent="claude_code")
    report = read_report(workspace)
    assert report["score"] == 220
    assert report["delivery_status"] == ("INCOMPLETE" if modify_report else "COMPLETE")
    assert report["exports"][0]["status"] == ("FAIL" if modify_report else "PASS")
    assert json.loads((workspace / "artifacts/solution.json").read_text())["delivered"]
    assert task_run_is_complete(workspace, "suite/protocol_fixture", "claude_code")
    if modify_report == "directory":
        state = workspace.parent / ".task-sessions" / workspace.name
        preserved = state / report["exports"][0]["invalid_report_artifact"]
        assert (preserved / "diagnostic.txt").read_text() == "exporter output"


def test_exporter_cannot_deliver_with_changed_protected_harness(tmp_path):
    code = '''import pathlib,os
target=pathlib.Path('evaluate.py')
target.write_text(target.read_text()+'\\n# exporter modified harness\\n')
output=pathlib.Path(os.environ['ARENA_EXPORT_PATH'])
output.parent.mkdir(parents=True,exist_ok=True)
output.write_text('{}')
'''
    path = package(tmp_path, exporter=code)
    _, workspace = run(tmp_path, path, lambda **_: None)
    report = read_report(workspace)
    assert report["pass_correctness"] and report["score"] == 220
    assert not report["candidate_accepted"]
    assert report["delivery_status"] == "INCOMPLETE"
    assert report["exports"][0]["protected_state_unchanged"] is False


def test_quality_loop_api_returns_exact_scored_report_and_rechecks_new_candidate(tmp_path):
    session = create(tmp_path)
    assert session.validate_initial().accepted
    config = {"agent": {"template": "quality_loop"}}
    first = evaluate_task_session(session, eval_config=config)
    assert first == read_report(session.workspace)
    assert first["score"] == 220
    (session.workspace / "kernel.py").write_text("def compute(): return -1\n")
    second = evaluate_task_session(session, eval_config=config)
    assert second == read_report(session.workspace)
    assert second["score"] == 20
    assert second["baseline_correctness"]["status"] == "PASS"


def test_exporter_cannot_deliver_a_different_candidate_than_the_evaluated_one(tmp_path):
    code = r'''import pathlib
pathlib.Path('kernel.py').write_text('def compute(): return 999\n')
raise RuntimeError('failed before writing the artifact')
'''
    path = package(tmp_path, exporter=code)
    _, workspace = run(tmp_path, path, lambda **_: None)
    report = read_report(workspace)
    assert report["pass_correctness"] and report["score"] == 220
    assert report["evaluated_candidate_sources"]["kernel.py"]
    assert report["delivery_status"] == "INCOMPLETE"
    assert not report["candidate_accepted"]
    assert report["exports"][0]["candidate_unchanged"] is False


@pytest.mark.parametrize("semantic_pass", [True, False])
def test_real_validator_launcher_uses_initial_evidence_and_semantic_gate(tmp_path, monkeypatch, semantic_pass):
    import importlib
    from datetime import datetime, timezone
    from agents.task_validator.report_schema import HARD_BENCHMARK_REVIEW_FIELDS, ADVISORY_BENCHMARK_REVIEW_FIELDS
    from agents.task_validator.report_v2 import DRAFT_FILENAME, SEMANTIC_CHECKS

    launcher = importlib.import_module("agents.task_validator.launch_agent")

    def backend(prompt, workspace, *_args, **_kwargs):
        # Exercise the real generated prompt and finalizers; only the model's
        # semantic judgment is a fixture. Never count this as actual LLM/GPU validation.
        raw = yaml.safe_load(prompt.rsplit("```yaml\n", 1)[1].split("```", 1)[0])
        raw["validation_timestamp"] = datetime.now(timezone.utc).isoformat()
        raw["overall_status"] = "PASS"
        for name in ("source_files_exist", "target_symbols_found", *SEMANTIC_CHECKS):
            raw["checks"][name].update(status="PASS", details="CPU fixture review",
                                       evidence=[{"path": "evaluate.py", "finding": "Fixture reviewed"}])
        raw["checks"]["benchmark_integrity"].update({key: True for key in (
            *HARD_BENCHMARK_REVIEW_FIELDS, *ADVISORY_BENCHMARK_REVIEW_FIELDS)})
        raw["checks"]["harness_integrity"].update(guard_coverage_reviewed=True, editable_targets_preserved=True)
        if not semantic_pass:
            raw["checks"]["correctness_implementation_review"]["status"] = "FAIL"
        (Path(workspace) / DRAFT_FILENAME).write_text(yaml.safe_dump(raw))
        return launcher.BackendResult(output="CPU fixture review", returncode=0, timed_out=False)

    monkeypatch.setattr(launcher, "_launch_codex", backend)
    path = package(tmp_path, empty=True, provided=3)
    complete, workspace = run(tmp_path, path, launcher.launch_agent, agent="task_validator")
    report = yaml.safe_load((workspace / "validation_report.yaml").read_text())
    assert complete
    assert report["overall_status"] == ("PASS" if semantic_pass else "FAIL"), report["validation_errors"]
    assert report["task_name"] == "suite/protocol_fixture"
    assert report["checks"]["correctness"]["status"] == "PASS"
    assert report["candidate_initial_checks"]["correctness"]["status"] == "SKIP"
    assert not (workspace / "task_result.yaml").exists()
