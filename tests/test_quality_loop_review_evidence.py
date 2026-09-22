"""Real CPU action processes; synthetic timings are not GPU qualification."""
import copy
import hashlib
import json
import logging
from pathlib import Path
import re
import shutil
import sys

import pytest
import yaml

from agents.quality_loop.runtime import EvaluationSession
from src.task_protocol import merge_command_results, parse_command_result
from src.task_session import TaskSession
from src.task_spec import load_task_spec
from tests.test_quality_loop_v2 import make_task, workflow


@pytest.fixture(autouse=True)
def cpu_runtime(monkeypatch):
    monkeypatch.setattr("src.task_runtime._runtime_identity", lambda: {
        "gpu_arch": "gfx950", "gpu_name": "CPU protocol fixture, no GPU"})
    monkeypatch.setenv("AGENT_KERNEL_ARENA_PYTHON", sys.executable)


def evaluated(tmp_path, *, state="unimplemented", wrong=False):
    workspace = make_task(tmp_path / "workspace", state=state,
                          baseline="provided" if state == "unimplemented" else "initial_candidate")
    spec = load_task_spec(workspace / "config.yaml", task_id="suite/nested/task")
    session = TaskSession.create(spec, workspace, tmp_path / "framework")
    adapter = EvaluationSession(session, eval_config={}, logger=logging.getLogger(__name__))
    adapter.prepare()
    shutil.copytree(workspace / "source", workspace / ".quality_loop_original_sources")
    (workspace / "source/kernel.py").write_text(
        "def kernel(x):\n    return " + ("3*x\n" if wrong else "x+x\n"))
    result = adapter.evaluate_candidate()
    return adapter, result


def read_locator(locator):
    path = Path(locator["path"])
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == locator["sha256"]
    return json.loads(data)


class Reviewer:
    def __init__(self, *, accepted=True, mutate=None):
        self.accepted, self.mutate = accepted, mutate
        self.calls = 0

    def run(self, prompt, workspace, *, role):
        self.calls += 1
        assert role == "reviewer"
        match = re.search(r"Framework evidence index: `([^`]+)` \(SHA256 `([a-f0-9]{64})`\)", prompt)
        assert match, "Actual reviewer prompt must expose the external evidence"
        path = Path(match[1])
        self.index = read_locator({"path": str(path), "sha256": match[2]})
        assert not path.is_relative_to(workspace)
        assert "aggregate and need not embed" in prompt
        assert "presence does not require acceptance" in prompt
        assert "candidate-authored links" in prompt
        if self.mutate:
            self.mutate(path, self.index)
        (workspace / "quality_loop_review.yaml").write_text(yaml.safe_dump({
            "accepted": self.accepted, "logic_equivalent": self.accepted,
            "evidence_sufficient": self.accepted, "case_enhancement_needed": False,
            "case_rationale": "CPU transport fixture", "summary": "CPU transport fixture",
        }))


def review(tmp_path, adapter, result, backend):
    loop = workflow(tmp_path, reviewer_backend=backend)
    loop._sessions[adapter.session.workspace] = adapter
    return loop._review(adapter.session.spec.task_id, adapter.session.workspace, result)


@pytest.mark.parametrize("state", ["implemented", "unimplemented"])
def test_reviewer_can_follow_actual_context_and_all_final_action_envelopes(tmp_path, state):
    adapter, result = evaluated(tmp_path, state=state)
    backend = Reviewer()
    assert review(tmp_path, adapter, result, backend)["accepted"]
    index = backend.index
    assert index["task_id"] == adapter.session.spec.task_id
    assert index["candidate_sources"] == result["evaluated_candidate_sources"]
    assert set(index["candidate_sources"]) == {"source/kernel.py"}
    context = read_locator(index["contexts"]["agent_context.json"])
    assert context["manifest"]["role"] == "task"
    initial = read_locator(index["contexts"]["validation_context.json"])
    assert all(row["phase"] == "task_validation" for row in initial["actions"])
    assert initial["initial_validation"]["candidate_initial_state"] == state
    for action, locator in index["candidate_actions"].items():
        record = read_locator(locator["record"])
        assert record["invocation_id"] == locator["invocation_id"]
        assert record["phase"] == "candidate_evaluation"
        assert record["result"]["role"] == "candidate"
        parsed = merge_command_results(parse_command_result(
            command["stdout"], role="candidate", action=action, returncode=command["returncode"]
        ) for command in record["commands"])
        assert parsed.to_mapping() == record["result"]
        assert [row["test_case_id"] for row in parsed.cases] == ([] if action == "compile" else ["1", "3"])
        assert record["commands"][0]["argv"] == [sys.executable, "runner.py", "candidate", action]
    assert yaml.safe_load(Path(index["result"]["path"]).read_text()) == result


@pytest.mark.parametrize("tamper", ["stdout", "argv", "result", "missing", "duplicate", "symlink",
                                    "context", "source", "report", "foreign_report"])
def test_candidate_cannot_substitute_or_forge_review_evidence(tmp_path, tamper):
    adapter, result = evaluated(tmp_path)
    session = adapter.session
    path = next(session.state_directory.glob("action-*-candidate-correctness.json"))
    record = json.loads(path.read_text())
    if tamper in ("stdout", "argv", "result"):
        if tamper == "stdout": record["commands"][0]["stdout"] = "fabricated PASS"
        if tamper == "argv": record["commands"][0]["argv"] = ["true"]
        if tamper == "result": record["result"]["cases"] = []
        path.write_text(json.dumps(record))
    elif tamper == "missing": path.unlink()
    elif tamper == "duplicate": shutil.copy2(path, session.state_directory / "action-9999-copy.json")
    elif tamper == "symlink":
        copied = tmp_path / "copied.json"
        shutil.copy2(path, copied)
        path.unlink(); path.symlink_to(copied)
    elif tamper == "context":
        session.agent_context_path.write_text('{"manifest": "fake"}')
    elif tamper == "source":
        (session.workspace / "source/kernel.py").write_text("def kernel(x): return 99\n")
    else:
        result = copy.deepcopy(result)
        if tamper == "foreign_report": result["task_name"] = "other/task"
        else: result["evidence_path"] = "candidate-authored-fake.json"
        (session.workspace / "task_result.yaml").write_text(yaml.safe_dump(result))
    backend = Reviewer()
    with pytest.raises((RuntimeError, OSError, ValueError)):
        review(tmp_path, adapter, result, backend)
    assert backend.calls == 0


@pytest.mark.parametrize("target", ["action", "context", "index", "baseline"])
def test_reviewer_cannot_mutate_external_evidence_or_baseline(tmp_path, target):
    adapter, result = evaluated(tmp_path)
    def mutate(path, index):
        if target == "action": path = Path(index["candidate_actions"]["performance"]["record"]["path"])
        elif target == "context": path = Path(index["contexts"]["agent_context.json"]["path"])
        elif target == "baseline": path = adapter.session.baseline_workspace / "source/kernel.py"
        path.write_text("changed\n")
    backend = Reviewer(mutate=mutate)
    with pytest.raises(RuntimeError, match="[Mm]odified protected|Frozen baseline"):
        review(tmp_path, adapter, result, backend)
    assert backend.calls == 1


def test_failed_correctness_is_exposed_and_reviewer_cannot_override_it(tmp_path):
    adapter, result = evaluated(tmp_path, wrong=True)
    backend = Reviewer(accepted=True)
    verdict = review(tmp_path, adapter, result, backend)
    assert not verdict["accepted"] and not verdict["evidence_sufficient"]
    assert backend.index["candidate_actions"]["correctness"]["status"] == "FAIL"
    assert backend.index["candidate_actions"]["performance"] == {"status": "NO_COMPLETED_ACTION", "record": None}


def test_existing_success_records_are_not_reused_after_later_framework_failure(tmp_path, monkeypatch):
    adapter, previous = evaluated(tmp_path)
    monkeypatch.setattr("src.task_runtime._runtime_identity", lambda: {"gpu_arch": "gfx942"})
    result = adapter.evaluate_candidate()
    assert not result["pass_compilation"] and previous["pass_correctness"]
    assert "runtime changed" in result["framework_error"]
    backend = Reviewer()
    assert not review(tmp_path, adapter, result, backend)["accepted"]
    assert all(row["status"] == "NO_COMPLETED_ACTION" for row in backend.index["candidate_actions"].values())


def test_completed_evidence_does_not_override_a_negative_review(tmp_path):
    adapter, result = evaluated(tmp_path)
    backend = Reviewer(accepted=False)
    assert result["pass_correctness"]
    assert not review(tmp_path, adapter, result, backend)["accepted"]


def test_later_evaluation_locates_only_its_own_actions(tmp_path):
    adapter, previous = evaluated(tmp_path)
    first = adapter.review_evidence(previous)
    first_index = json.loads(first.path.read_text())
    result = adapter.evaluate_candidate()
    evidence = adapter.review_evidence(result)
    current = json.loads(evidence.path.read_text())
    assert first.path.exists() and first.path != evidence.path
    for action, row in current["candidate_actions"].items():
        assert row["invocation_id"] != first_index["candidate_actions"][action]["invocation_id"]
        record = read_locator(row["record"])
        assert record["invocation_id"] == adapter.session.results[
            ("candidate_evaluation", "candidate", action)].invocation_id


def test_failed_index_persistence_never_starts_reviewer(tmp_path, monkeypatch):
    adapter, result = evaluated(tmp_path)
    def fail(_):
        raise OSError("injected receipt fsync error")
    monkeypatch.setattr("agents.quality_loop.review_evidence.os.fsync", fail)
    backend = Reviewer()
    with pytest.raises(OSError, match="injected receipt fsync"):
        review(tmp_path, adapter, result, backend)
    assert backend.calls == 0


def test_report_without_controller_session_is_not_reviewable(tmp_path):
    adapter, result = evaluated(tmp_path)
    backend = Reviewer()
    with pytest.raises(RuntimeError, match="controller session"):
        workflow(tmp_path, reviewer_backend=backend)._review(
            adapter.session.spec.task_id, adapter.session.workspace, result)
    assert backend.calls == 0
