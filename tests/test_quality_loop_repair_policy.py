"""Maintenance regressions: preserve scoring and independently close findings."""
from copy import deepcopy
import logging

import pytest
import yaml

from agents.quality_loop.config import QualityLoopConfig
from agents.quality_loop.orchestrator import QualityLoop
from agents.quality_loop.repair_policy import RepairPolicy, RepairRejected, blocking_findings
from src.task_spec import load_task_spec
from tests.test_quality_loop import StubQualityLoop, FakePublisher, attach_state, make_task


CASE = dict(test_case_id="a", shape=[4], dtype="float32", params={"seed": 7},
            checks=["correctness", "performance"])


def test_repair_policy_accepts_only_additional_unscored_cases(tmp_path):
    task = make_task(tmp_path)
    spec = load_task_spec(task / "config.yaml", task_id="hip2hip/sample")
    policy = RepairPolicy.capture(spec, [CASE])
    extra = dict(CASE, test_case_id="zero", checks=["correctness"])
    policy.verify(spec, [CASE, extra])
    for changed in ([], [dict(CASE, params={"seed": 8})],
                    [dict(CASE, checks=["correctness"])], [CASE, dict(extra, checks=["performance"]) ]):
        with pytest.raises((RepairRejected, RuntimeError)):
            policy.verify(spec, changed)


def test_repair_cannot_change_samples_method_cache_or_task_spec(tmp_path):
    task = make_task(tmp_path)
    spec = load_task_spec(task / "config.yaml", task_id="hip2hip/sample")
    row = {**CASE, "benchmark_method": "cuda_graph", "metadata": {
        "benchmark_samples": 100, "benchmark_cache_policy": "warm"}}
    report = {"checks": {"performance": {"result": {"cases": [row]}}}}
    policy = RepairPolicy.capture(spec, [CASE], report)
    for key, value in (("benchmark_samples", 1), ("benchmark_cache_policy", "cold")):
        proposed = deepcopy(report)
        proposed["checks"]["performance"]["result"]["cases"][0]["metadata"][key] = value
        with pytest.raises(RepairRejected, match="benchmark policy"):
            policy.verify(spec, [CASE], proposed)
    config = spec.to_mapping(); config["candidate"]["language"] = "triton"
    from src.task_spec import TaskSpec
    with pytest.raises(RepairRejected, match="specification"):
        policy.verify(TaskSpec.from_mapping(config, task_id=spec.task_id), [CASE], report)


@pytest.mark.parametrize('replacement', [True, 1.0])
def test_repair_policy_retains_json_types_in_identity_and_timing(tmp_path, replacement):
    task = make_task(tmp_path)
    spec = load_task_spec(task / 'config.yaml', task_id='hip2hip/sample')
    case = dict(CASE, params={'seed': 1})
    report = {'checks': {'performance': {'result': {'cases': [
        {**case, 'benchmark_method': 'cuda_graph', 'metadata': {'benchmark_samples': 1}}
    ]}}}}
    policy = RepairPolicy.capture(spec, [case], report)
    with pytest.raises(RepairRejected, match='original case'):
        policy.verify(spec, [dict(case, params={'seed': replacement})], report)
    modified = deepcopy(report)
    modified['checks']['performance']['result']['cases'][0]['metadata']['benchmark_samples'] = replacement
    with pytest.raises(RepairRejected, match='benchmark policy'):
        policy.verify(spec, [case], modified)


def test_repair_mode_configuration_is_bounded_and_explicit():
    cfg = QualityLoopConfig.from_dict({"target_gpu_model": "MI355X", "quality_loop": {
        "mode": "repair_only", "max_repair_attempts": 3,
        "repair_warnings": ["correctness_implementation_review"]}})
    assert cfg.optimization_iterations == 0 and cfg.mode == "repair_only"
    for value in (0, 4, True, "2"):
        with pytest.raises(ValueError):
            QualityLoopConfig.from_dict({"target_gpu_model": "MI355X", "quality_loop": {
                "max_repair_attempts": value}})
    with pytest.raises(ValueError):
        QualityLoopConfig.from_dict({"target_gpu_model": "MI355X", "quality_loop": {
            "repair_warnings": ["unknown"]}})


def test_findings_keep_independent_ids_and_selected_warnings():
    report = {"checks": {"correctness": {"status": "FAIL", "details": "wrong"},
                         "benchmark_integrity": {"status": "WARN", "details": "missing replay"}}}
    assert len(blocking_findings(report)) == 1
    assert len(blocking_findings(report, ("benchmark_integrity",))) == 2
    previous = blocking_findings(report)
    report["checks"]["correctness"]["details"] = "another defect"
    assert previous.keys().isdisjoint(blocking_findings(report))


class ChangingBackend:
    def __init__(self):
        self.calls = 0

    def run(self, prompt, workspace, *, role):
        assert role == "repair"
        self.calls += 1
        (workspace / "check.py").write_text(f"CONTROL = {self.calls}\n")


def test_repair_only_closes_all_findings_without_optimizer_or_fake_speedup(tmp_path, monkeypatch):
    task = make_task(tmp_path / "worktree")
    backend = ChangingBackend()
    fail = {"overall_status": "FAIL", "checks": {"correctness_implementation_review": {
        "status": "FAIL", "details": "missing zero input", "evidence": []}}}
    loop = StubQualityLoop(tmp_path, QualityLoopConfig.from_dict({
        "target_gpu_model": "MI355X", "quality_loop": {"mode": "repair_only", "max_repair_attempts": 2}}),
        logger=logging.getLogger(__name__), backend=backend, publisher=FakePublisher(),
        reports=[fail, fail, {"overall_status": "PASS", "checks": {}}])
    attach_state(loop, tmp_path, task.parents[2])
    monkeypatch.setattr(loop, "_optimize_once", lambda *a: pytest.fail("repair-only optimized"))
    loop._process_task("hip2hip/sample", task)
    assert backend.calls == 2
    assert (task / "check.py").read_text() == "CONTROL = 2\n"
    record = loop.state.task("hip2hip/sample")
    assert record["state"] == "completed" and record["speedups"] == []
    assert len(record["reviewer"]["resolved_findings"]) == 1


@pytest.mark.parametrize("tamper", [False, True])
def test_repair_review_cannot_omit_findings_or_mutate_original(tmp_path, tamper):
    original = make_task(tmp_path / "original")
    workspace = tmp_path / "workspace"; workspace.mkdir()
    (workspace / "kernel.py").write_text("valid source")
    class Reviewer:
        def run(self, prompt, work, *, role):
            if tamper:
                (original / "kernel.py").write_text("changed")
            (work / "quality_loop_repair_review.yaml").write_text(yaml.safe_dump(dict(
                accepted=True, contract_preserved=True, scoring_preserved=True,
                resolved_findings=[], summary="unsupported acceptance")))
    loop = QualityLoop(tmp_path, QualityLoopConfig(), logger=logging.getLogger(__name__),
                       reviewer_backend=Reviewer())
    if tamper:
        with pytest.raises(RuntimeError, match="protected files"):
            loop._review_repair("hip2hip/sample", original, workspace, {"known": {}})
    else:
        assert loop._review_repair("hip2hip/sample", original, workspace, {"known": {}})["accepted"] is False
