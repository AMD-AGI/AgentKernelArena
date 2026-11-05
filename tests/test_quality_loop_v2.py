"""quality_loop over schema-v3 packages and the QA auditor (CPU only).

The audit tests run the real materialization, TaskSession and
src.task_run.audit_task_session with a fake auditor that finalizes reports
through agents.qa_auditor.report. On CPU the initial GPU actions fail; that is
recorded evidence, never GPU validation.
"""
from datetime import datetime, timedelta, timezone
import json
import logging
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest
import yaml

from agents.qa_auditor.report import (
    COMPLETION_MARKER_FILENAME, REPORT_FILENAME, build_report, write_report,
)
from agents.quality_loop.config import QualityLoopConfig
from agents.quality_loop.filesystem import snapshot_tree
from agents.quality_loop.orchestrator import QualityLoop, _solution_paths, _task_slug, difficulty_is_easy
from agents.quality_loop.prompts import optimizer_prompt
from agents.quality_loop.runtime import InitialValidationRejected
from agents.quality_loop.state import stable_fingerprint
from src.task_contract import load_task_contract
from src.task_materialization import load_materialization_record
from src.task_spec import TaskConfigError, load_task_spec
from tests.test_quality_loop import CASES, TASK_ID, TASK_YAML, write_package


LOGGER = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def cpu_runtime(monkeypatch):
    monkeypatch.setattr("src.task_runtime._runtime_identity", lambda: {
        "gpu_arch": "gfx950", "gpu_name": "CPU protocol fixture, no GPU"})
    if sys.platform == "darwin":
        # macOS cp has no --reflink; use the framework's pure-Python copy.
        import types
        import src.task_materialization as materialization
        proxy = types.SimpleNamespace(**{name: getattr(shutil, name) for name in dir(shutil)
                                         if not name.startswith("__")})
        proxy.which = lambda name, *args, **kwargs: None if name == "cp" else shutil.which(name, *args, **kwargs)
        monkeypatch.setattr(materialization, "shutil", proxy)


def make_task(root, **kwargs):
    return write_package(Path(root), **kwargs)


def workflow(root, **kwargs):
    config = kwargs.pop("config", None) or QualityLoopConfig.from_dict({"target_gpu_model": "MI355X"})
    return QualityLoop(Path(root), config, logger=LOGGER, **kwargs)


def finalize(workspace, settings, *, findings=(), task_id=None, request_id=None, age=0):
    """The auditor's real report and completion marker for this workspace."""
    record = load_materialization_record(Path(workspace))
    report = build_report(task_id=task_id or settings["_task_id"], findings=list(findings),
                          task_package_sha256=record["task_package_sha256"], target_language="triton",
                          request_id=request_id or settings["_task_validation_request_id"])
    report["generated_at"] = (datetime.now(timezone.utc) - timedelta(seconds=age)).isoformat()
    write_report(Path(workspace), report)
    return report


CONTROL = {"id": "QA-CONTROLS", "severity": "blocking", "title": "controls", "detail": "zero passes",
           "evidence": {}, "fix": "strengthen"}


def test_plan_uses_contract_platform_and_skips_nested_packages(tmp_path):
    raw = dict(TASK_YAML, platform_support={"required_arch": ["gfx942"]})
    task = make_task(tmp_path / "tasks/elementwise/deep/name", task_yaml=raw)
    # A task.yaml below a task is not another task.
    nested = task / "docs"
    nested.mkdir()
    (nested / "task.yaml").write_text("unrelated: true\n")
    loop = workflow(tmp_path)
    assert list(loop.discover_tasks()) == ["elementwise/deep/name"]
    plan = loop.plan()
    assert plan["platform_deferred"] == ["elementwise/deep/name"] and plan["runnable"] == []
    assert _task_slug("a/b") != _task_slug("a__b")


def test_invalid_v3_package_is_rejected_instead_of_adapted(tmp_path):
    task = tmp_path / "tasks/old"
    task.mkdir(parents=True)
    (task / "task.yaml").write_text("schema_version: 3\ntask_type: torch2hip\n")
    with pytest.raises(TaskConfigError):
        workflow(tmp_path).plan()


def test_optimizer_prompt_describes_the_v3_contract(tmp_path):
    task = make_task(tmp_path / "task")
    spec = load_task_spec(task / "task.yaml", task_id=TASK_ID, target_language="hip")
    text = optimizer_prompt("contract", TASK_ID, spec)
    assert "`hip`" in text and "`triton`" in text and "unimplemented" in text
    assert "_aka_eval.evaluate candidate correctness" in text and "solution/" in text
    assert "schema_version: 2" not in text and "initial_language" not in text


def test_explicit_budgets_and_auditor_independence(tmp_path):
    config = QualityLoopConfig.from_dict({"target_gpu_model": "MI355X", "target_language": "hip",
                                          "quality_loop": {
        "backend": {"model": "gpt-6-astra", "effort": "high"},
        "auditor": {"name": "claude_code", "model": None, "effort": "max", "timeout_seconds": 900},
    }})
    loop = QualityLoop(tmp_path, config, logger=LOGGER)
    audit = loop._eval_config(task_id="suite/task", auditor=True)
    assert audit["agent"] == {"template": "qa_auditor", "backend": "claude_code", "model": None,
                              "effort": "max", "timeout_seconds": 900}
    assert audit["_task_id"] == "suite/task" and "target_language" not in audit
    optimize = loop._eval_config(task_id="suite/task", language="hip")
    assert optimize["agent"]["model"] == "gpt-6-astra" and optimize["target_language"] == "hip"
    assert loop._optimization_language("suite/task", make_task(tmp_path / "t")) == "hip"
    default = workflow(tmp_path)
    assert default._optimization_language(TASK_ID, tmp_path / "t") == "triton"


def test_missing_tool_evidence_cannot_satisfy_easy_gate():
    config = QualityLoopConfig.from_dict({"target_gpu_model": "MI355X",
                                          "evaluation_tools": {"enabled": ["gpu_asan"]}})
    result = {"pass_compilation": True, "pass_correctness": True, "benchmark_method_consistent": True,
              "valid_baseline_cases": 2, "valid_optimized_cases": 2}
    review = {"accepted": True, "logic_equivalent": True, "evidence_sufficient": True}
    assert not difficulty_is_easy(speedups=[6, 6, 6], result=result, review=review, config=config)
    result.update(pass_tool_gate=True, tool_policy_satisfied=True)
    assert difficulty_is_easy(speedups=[6, 6, 6], result=result, review=review, config=config)


def test_copy_excludes_framework_files_and_archive_preserves_reports(tmp_path):
    source = make_task(tmp_path / "source")
    for name in (REPORT_FILENAME, COMPLETION_MARKER_FILENAME, "config.yaml", "task_result.yaml",
                 "solution/kernel.py", "_aka_eval/evaluate.py"):
        (source / name).parent.mkdir(parents=True, exist_ok=True)
        (source / name).write_text("generated")
    QualityLoop._copy_task(source, tmp_path / "copy")
    assert sorted(snapshot_tree(tmp_path / "copy")) == [
        "README.md", "baseline/kernel.py", "reference.py", "task.yaml", "workloads.json"]
    QualityLoop._reset_path(source / REPORT_FILENAME)
    saved = list((source / ".quality_loop_history").glob(REPORT_FILENAME + "-*"))
    assert len(saved) == 1 and saved[0].read_text() == "generated"


def test_solution_paths_cover_nested_tree_and_reject_escape(tmp_path):
    workspace = tmp_path / "workspace"
    (workspace / "solution/helpers").mkdir(parents=True)
    (workspace / "solution/kernel.py").write_text("")
    (workspace / "solution/helpers/tile.py").write_text("")
    (workspace / "solution/__pycache__").mkdir()
    (workspace / "solution/__pycache__/kernel.cpython-312.pyc").write_text("")
    assert _solution_paths(workspace) == ("solution/helpers/tile.py", "solution/kernel.py")
    (workspace / "solution/escape.py").symlink_to(tmp_path / "outside.py")
    with pytest.raises(TaskConfigError):
        _solution_paths(workspace)


@pytest.mark.parametrize("failure", ["missing", "raw", "wrong_request", "wrong_task", "stale",
                                     "launcher_error", "tampered", "modified_task", "modified_package"])
def test_audit_cannot_pass_missing_stale_or_untrusted_evidence(tmp_path, failure):
    task = make_task(tmp_path / "task")

    def auditor(settings, task_config_dir, workspace):
        workspace = Path(workspace)
        if failure == "missing":
            return
        if failure == "launcher_error":
            raise RuntimeError("auditor crashed")
        if failure == "raw":
            (workspace / REPORT_FILENAME).write_text("status: PASS\n")
            return
        finalize(workspace, settings, task_id="other/task" if failure == "wrong_task" else None,
                 request_id="0" * 32 if failure == "wrong_request" else None,
                 age=120 if failure == "stale" else 0)
        if failure == "tampered":
            (workspace / REPORT_FILENAME).write_text("status: PASS\nfindings: []\n")
        if failure == "modified_task":
            (workspace / "reference.py").write_text("def reference(*, x):\n    return x\n")
        if failure == "modified_package":
            (Path(task_config_dir).parent / "README.md").write_text("changed during the audit\n")

    reason = {"stale": "stale", "modified_task": "modified task files",
              "modified_package": "exact task package"}.get(failure, "did not run to completion")
    with pytest.raises(RuntimeError, match=reason):
        workflow(tmp_path, auditor_launcher=auditor)._audit(TASK_ID, task, tmp_path / "audit")


def test_audit_uses_stable_identity_audit_language_and_finalized_report(tmp_path):
    task = make_task(tmp_path / "arbitrary-copy-name")
    seen = []

    def auditor(settings, task_config_dir, workspace):
        assert str(task_config_dir).endswith(f"/tasks/{TASK_ID}/task.yaml")
        assert settings["agent"]["template"] == "qa_auditor"
        config = yaml.safe_load((Path(workspace) / "config.yaml").read_text())
        assert config["contract"]["target_language"] == "triton"
        assert config["candidate"]["initial_state"] == "implemented"
        seen.append(settings["_task_id"])
        finalize(workspace, settings)

    loop = workflow(tmp_path, auditor_launcher=auditor)
    workspace, report = loop._audit(TASK_ID, task, tmp_path / "audit")
    assert report["status"] == "PASS" and seen == [TASK_ID]
    [evidence] = loop._audit_evidence
    assert evidence["target_language"] == "triton" and evidence["status"] == "PASS"
    assert (workspace / REPORT_FILENAME).is_file()


def test_blocking_finding_is_a_task_judgment_not_an_operational_failure(tmp_path):
    task = make_task(tmp_path / "task")
    loop = workflow(tmp_path, auditor_launcher=lambda s, c, w: finalize(w, s, findings=[CONTROL]))
    _, report = loop._audit(TASK_ID, task, tmp_path / "audit")
    assert report["status"] == "FAIL" and report["checks"]["QA-CONTROLS"] == "FAIL"


@pytest.mark.parametrize("corruption", ["none", "source", "report", "marker", "record", "escape", "rubric"])
def test_resume_requires_unchanged_task_and_finalized_evidence(tmp_path, monkeypatch, corruption):
    task = make_task(tmp_path / "task")
    loop = workflow(tmp_path, auditor_launcher=lambda s, c, w: finalize(w, s))
    workspace, _ = loop._audit(TASK_ID, task, tmp_path / "audit")
    record = {"state": "completed", "accepted_fingerprint": stable_fingerprint(snapshot_tree(task)),
              "audit_evidence": loop._audit_evidence}
    loop.state = SimpleNamespace(task=lambda task_id: record)
    if corruption == "source":
        (task / "README.md").write_text("changed\n")
    if corruption == "report":
        (workspace / REPORT_FILENAME).write_text("status: PASS\n")
    if corruption == "marker":
        (workspace / COMPLETION_MARKER_FILENAME).unlink()
    if corruption == "record":
        record["audit_evidence"] = [{}]
    if corruption == "escape":
        record["audit_evidence"][0]["workspace"] = "../elsewhere"
    if corruption == "rubric":
        monkeypatch.setattr("agents.qa_auditor.report.RUBRIC_VERSION", "a-later-rubric")
    assert loop._terminal_evidence_current(TASK_ID, task) is (corruption == "none")


def optimized(tmp_path, spec, files):
    workspace = tmp_path / "optimized"
    for relative, text in files.items():
        (workspace / relative).parent.mkdir(parents=True, exist_ok=True)
        (workspace / relative).write_text(text)
    return workspace


@pytest.mark.parametrize("audit_status", ["PASS", "FAIL"])
def test_same_language_promotion_replaces_the_baseline_and_records_provenance(tmp_path, monkeypatch, audit_status):
    original = make_task(tmp_path / "original")
    (original / "baseline/LICENSE").write_text("upstream license\n")
    loop = workflow(tmp_path)
    candidate = tmp_path / "candidate"
    loop._copy_task(original, candidate)
    spec = load_task_spec(original / "task.yaml", task_id=TASK_ID, target_language="triton")
    workspace = optimized(tmp_path, spec, {"solution/kernel.py": "def run(*, x, out):\n    pass\n",
                                           "solution/tiles.py": "BLOCK = 256\n"})
    loop._sessions[workspace] = SimpleNamespace(session=SimpleNamespace(spec=spec))
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    before = snapshot_tree(candidate)
    gates = []
    monkeypatch.setattr(loop, "_dual_correctness_gate", lambda *a, **k: gates.append(k) or True)
    monkeypatch.setattr(loop, "_audit", lambda *a: (None, {"status": audit_status}))
    accepted, _ = loop._promote_baseline(TASK_ID, original, candidate, workspace, artifacts)
    assert accepted is (audit_status == "PASS")
    assert gates[0]["optimized_workspace"] == workspace
    if accepted:
        assert (candidate / "baseline/kernel.py").read_text() == "def run(*, x, out):\n    pass\n"
        assert (candidate / "baseline/tiles.py").read_text() == "BLOCK = 256\n"
        assert (candidate / "baseline/LICENSE").read_text() == "upstream license\n"
        provenance = load_task_contract(candidate / "task.yaml", task_id=TASK_ID).raw["baseline"]["provenance"]
        assert "quality_loop" in provenance["source"] and len(provenance["revision"]) == 64
        assert provenance["license"] == "Apache-2.0" and "arena-tests" in provenance["modifications"]
    else:
        assert snapshot_tree(candidate) == before


def test_cross_language_candidate_is_not_promoted(tmp_path):
    original = make_task(tmp_path / "original")
    spec = load_task_spec(original / "task.yaml", task_id=TASK_ID, target_language="hip")
    workspace = optimized(tmp_path, spec, {"solution/kernel.hip": "// hip\n"})
    loop = workflow(tmp_path)
    loop._sessions[workspace] = SimpleNamespace(session=SimpleNamespace(spec=spec))
    before = snapshot_tree(original)
    accepted, reason = loop._promote_baseline(TASK_ID, original, original, workspace, tmp_path / "a")
    assert accepted is False and "same-language" in reason
    assert snapshot_tree(original) == before


def test_stale_reviewer_decision_rejected(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "quality_loop_review.yaml").write_text("accepted: true")
    with pytest.raises(RuntimeError, match="old reviewer"):
        workflow(tmp_path)._review("suite/task", workspace, {})


def test_session_cannot_reuse_old_score_file(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    original = "task_name: suite/task\npass_correctness: true\n"
    (workspace / "task_result.yaml").write_text(original)
    loop = workflow(tmp_path)
    loop._sessions[workspace] = SimpleNamespace(evaluate_candidate=lambda: yaml.safe_load(original))
    with pytest.raises(RuntimeError, match="fresh task_result"):
        loop._evaluate_session(workspace)
    assert next((workspace / ".quality_loop_history").iterdir()).read_text() == original


def test_each_measurement_uses_same_session_and_checks_identity(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    calls = []

    def evaluate():
        result = {"task_name": "suite/task" if not calls else "wrong/task", "speedup_ratio": 2.0}
        calls.append(result)
        (workspace / "task_result.yaml").write_text(yaml.safe_dump(result))
        return result

    loop = workflow(tmp_path)
    loop._sessions[workspace] = SimpleNamespace(evaluate_candidate=evaluate)
    loop._session_task_ids[workspace] = "suite/task"
    assert loop._evaluate_session(workspace)["task_name"] == "suite/task"
    with pytest.raises(RuntimeError, match="different task identity"):
        loop._evaluate_session(workspace)
    assert len(calls) == 2


class FakeSession:
    """The EvaluationSession surface; the real one is in test_quality_loop_review_evidence."""

    def __init__(self, *, spec, workspace, artifact_dir, eval_config, logger, reject=False, passes=True):
        self.spec, self.workspace, self.eval_config = spec, Path(workspace), eval_config
        self.reject, self.passes = reject, passes
        self.session = SimpleNamespace(spec=spec)
        self.agent_context_path = Path(artifact_dir) / "agent_context.json"
        self.baseline_cases = [{"test_case_id": "n1024"}]
        self.evaluations = 0

    def prepare(self):
        if self.reject:
            raise InitialValidationRejected("original baseline fails the proposed case")

    def check_candidate(self):
        return self.passes

    def evaluate_candidate(self):
        self.evaluations += 1
        result = {"task_name": self.spec.task_id, "pass_compilation": True, "pass_correctness": True,
                  "speedup_ratio": 1.0, "synthetic_cpu_fixture": True}
        (self.workspace / "task_result.yaml").write_text(yaml.safe_dump(result))
        return result


def proposed_cases():
    return CASES + [{"id": "n7", "axes": {"n": 7}, "checks": ["correctness"], "source": "odd tail"}]


@pytest.mark.parametrize("outcome", ["pass", "candidate_fails", "baseline_rejected"])
def test_dual_gate_checks_original_baseline_and_optimized_candidate_on_new_cases(tmp_path, outcome):
    original = make_task(tmp_path / "original")
    proposed = make_task(tmp_path / "proposed", cases=proposed_cases())
    (proposed / "baseline/kernel.py").write_text("# a promoted baseline the gate must not use\n")
    sessions = []

    def factory(**kwargs):
        sessions.append(FakeSession(**kwargs, reject=outcome == "baseline_rejected",
                                    passes=outcome != "candidate_fails"))
        return sessions[-1]

    spec = load_task_spec(original / "task.yaml", task_id=TASK_ID, target_language="hip")
    workspace = optimized(tmp_path, spec, {"solution/kernel.hip": "// optimized hip\n"})
    loop = workflow(tmp_path, session_factory=factory)
    loop._sessions[workspace] = SimpleNamespace(session=SimpleNamespace(spec=spec))
    assert loop._dual_correctness_gate(TASK_ID, original, proposed, tmp_path / "gate",
                                       optimized_workspace=workspace) is (outcome == "pass")
    [session] = sessions
    gate = session.workspace
    assert session.spec.target_language == "hip"
    assert (gate / "baseline/kernel.py").read_text() == (original / "baseline/kernel.py").read_text()
    assert [case["id"] for case in json.loads((gate / "workloads.json").read_text())["cases"]][-1] == "n7"
    if outcome != "baseline_rejected":
        assert (gate / "solution/kernel.hip").read_text() == "// optimized hip\n"


@pytest.mark.parametrize("edit", ["correctness_case", "performance_case", "readme", "changed_case"])
def test_case_enhancement_accepts_only_added_correctness_cases(tmp_path, monkeypatch, edit):
    original = make_task(tmp_path / "original")
    candidate = make_task(tmp_path / "candidate")

    class CaseBackend:
        def run(self, prompt, workspace, *, role):
            assert role == "case_enhancer" and "workloads.json" in prompt
            cases = proposed_cases()
            if edit == "performance_case":
                cases[-1]["checks"] = ["correctness", "performance"]
            if edit == "changed_case":
                cases[1]["axes"] = {"n": 5}
            if edit == "readme":
                (workspace / "README.md").write_text("rewritten\n")
            (workspace / "workloads.json").write_text(json.dumps({"cases": cases}))

    loop = workflow(tmp_path, backend=CaseBackend())
    monkeypatch.setattr(loop, "_dual_correctness_gate", lambda *a, **k: True)
    monkeypatch.setattr(loop, "_audit", lambda *a: (None, {"status": "PASS"}))
    before = snapshot_tree(candidate)
    accepted = loop._enhance_cases(TASK_ID, original, candidate, "missing odd tail",
                                   tmp_path / "artifacts", optimized_workspace=tmp_path / "optimized")
    assert accepted is (edit == "correctness_case")
    if accepted:
        assert json.loads((candidate / "workloads.json").read_text())["cases"][-1]["id"] == "n7"
        load_task_contract(candidate / "task.yaml", task_id=TASK_ID)
    else:
        assert snapshot_tree(candidate) == before


def test_optimizer_wiring_uses_shared_session_and_v3_prompt(tmp_path):
    pytest.importorskip("src.task_prompt")
    task = make_task(tmp_path / "task")
    sessions = []

    def factory(**kwargs):
        sessions.append(FakeSession(**kwargs))
        return sessions[-1]

    class Optimizer:
        def run(self, prompt, workspace, *, role):
            assert role == "optimizer"
            assert TASK_ID in prompt and "agent_context.json" in prompt
            assert "Target language: `hip`" in prompt and "unimplemented" in prompt
            assert "_aka_eval.evaluate candidate correctness" in prompt
            assert "AKA-UNIMPLEMENTED-TEMPLATE" in (workspace / "solution/kernel.hip").read_text()
            (workspace / "solution/kernel.hip").write_text("// candidate\n")

    config = QualityLoopConfig.from_dict({"target_gpu_model": "MI355X", "target_language": "hip"})
    loop = workflow(tmp_path, config=config, backend=Optimizer(), session_factory=factory)
    workspace, baseline_cases, result = loop._optimize_once(TASK_ID, task, tmp_path / "optimize")
    assert baseline_cases and result["synthetic_cpu_fixture"] is True
    assert sessions[0].spec.target_language == "hip" and sessions[0].eval_config["target_language"] == "hip"
    original = workspace / ".quality_loop_original_sources/solution/kernel.hip"
    assert "AKA-UNIMPLEMENTED-TEMPLATE" in original.read_text()
    loop._evaluate_session(workspace)
    assert sessions[0].evaluations == 2
    assert not (task / "solution").exists()
