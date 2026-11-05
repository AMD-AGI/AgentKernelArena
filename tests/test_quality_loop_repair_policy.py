"""Maintenance regressions: preserve scoring and independently close QA findings."""
from copy import deepcopy
import json
import logging

import pytest
import yaml

from agents.quality_loop.config import QualityLoopConfig
from agents.quality_loop.orchestrator import QualityLoop
from agents.quality_loop.repair_policy import (
    RepairPolicy, RepairRejected, blocking_findings, compare_tightened,
)
from tests.test_quality_loop import (
    CASES, TASK_ID, TASK_YAML, FakePublisher, StubQualityLoop, attach_state, finding, make_task,
    qa_report, write_package,
)


def package(tmp_path, name="task", **kwargs):
    return write_package(tmp_path / name, **kwargs)


def edit_yaml(task, change):
    raw = yaml.safe_load((task / "task.yaml").read_text())
    change(raw)
    (task / "task.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))


def edit_cases(task, change):
    cases = json.loads((task / "workloads.json").read_text())
    change(cases["cases"])
    (task / "workloads.json").write_text(json.dumps(cases))


EXTRA = {"id": "n7", "axes": {"n": 7}, "checks": ["correctness"], "source": "odd tail"}


def test_repair_policy_accepts_only_additional_correctness_cases(tmp_path):
    policy = RepairPolicy.capture(package(tmp_path), TASK_ID)
    policy.verify(package(tmp_path, "extra", cases=CASES + [EXTRA]))
    rejected = {
        "changed": [dict(CASES[0], axes={"n": 2})] + CASES[1:] + [EXTRA],
        "removed": CASES[1:4] + [dict(CASES[0], id="first")] + CASES[4:],
        "checks": CASES[:3] + [dict(CASES[3], checks=["correctness"])] + CASES[4:]
                  + [dict(EXTRA, id="p", axes={"n": 2048}, checks=["correctness", "performance"])],
        "scored_addition": CASES + [dict(EXTRA, checks=["correctness", "performance"])],
    }
    for name, cases in rejected.items():
        with pytest.raises(RepairRejected):
            policy.verify(package(tmp_path, name, cases=cases))


@pytest.mark.parametrize("change", [
    lambda raw: raw["interface"]["inputs"]["x"].update(dtype="float16"),
    lambda raw: raw["interface"]["inputs"]["x"]["distribution"].update(std=0.5),
    lambda raw: raw["baseline"]["provenance"].update(revision="v2"),
    lambda raw: raw.update(timeout_s=600),
    lambda raw: raw["operator"].update(name="scale2"),
])
def test_repair_cannot_change_the_scored_contract(tmp_path, change):
    policy = RepairPolicy.capture(package(tmp_path), TASK_ID)
    proposal = package(tmp_path, "proposal")
    edit_yaml(proposal, change)
    with pytest.raises(RepairRejected, match="scored task contract"):
        policy.verify(proposal)


def test_repair_cannot_change_baseline_sources_but_may_fix_docs_inputs_and_license(tmp_path):
    policy = RepairPolicy.capture(package(tmp_path), TASK_ID)
    proposal = package(tmp_path, "docs")
    edit_yaml(proposal, lambda raw: raw.update(description="Clarified: out = 2 * x, exactly."))
    (proposal / "README.md").write_text("Documents the boundary cases.\n")
    (proposal / "reference.py").write_text("def reference(*, x):\n    return 2 * x.double()\n")
    (proposal / "baseline/LICENSE").write_text("license text\n")
    policy.verify(proposal)
    (proposal / "baseline/kernel.py").write_text("def run(*, x, out):\n    pass\n")
    with pytest.raises(RepairRejected, match="vendored baseline"):
        policy.verify(proposal)


def test_invalid_proposal_is_a_rejected_repair(tmp_path):
    policy = RepairPolicy.capture(package(tmp_path), TASK_ID)
    proposal = package(tmp_path, "broken")
    edit_cases(proposal, lambda cases: cases.pop(0))
    with pytest.raises(RepairRejected, match="invalid task contract"):
        policy.verify(proposal)


@pytest.mark.parametrize("original,proposed,tighter", [
    ({"bound": "baseline_relative"}, {"bound": "baseline_relative", "k": 1.5}, True),
    ({"bound": "baseline_relative"}, {"bound": "baseline_relative", "baseline_min_sqnr_db": 30}, True),
    ({"bound": "baseline_relative"}, {"bound": "exact"}, True),
    ({"bound": "baseline_relative"}, {"bound": "baseline_relative", "k": 3}, False),
    ({"bound": "baseline_relative"}, {"bound": "baseline_relative", "max_violation_fraction": 0.001}, False),
    ({"bound": "baseline_relative"}, {"bound": "tolerance", "atol": 0, "rtol": 0}, False),
    ({"bound": "tolerance", "atol": 1e-3, "rtol": 1e-3}, {"bound": "tolerance", "atol": 1e-4, "rtol": 1e-3}, True),
    ({"bound": "sqnr", "min_db": 30}, {"bound": "sqnr", "min_db": 25}, False),
    ({"bound": "dot_product", "length": 4096}, {"bound": "dot_product", "length": 8192}, False),
])
def test_compare_bounds_may_only_tighten(tmp_path, original, proposed, tighter):
    assert compare_tightened(original, proposed) is tighter
    raw = deepcopy(TASK_YAML)
    raw["compare"] = {"out": original}
    policy = RepairPolicy.capture(package(tmp_path, task_yaml=raw), TASK_ID)
    proposal = package(tmp_path, "proposal", task_yaml=dict(raw, compare={"out": proposed}))
    if tighter:
        policy.verify(proposal)
    else:
        with pytest.raises(RepairRejected, match="error bound"):
            policy.verify(proposal)


def test_repair_mode_configuration_is_bounded_and_explicit():
    cfg = QualityLoopConfig.from_dict({"target_gpu_model": "MI355X", "quality_loop": {
        "mode": "repair_only", "max_repair_attempts": 3, "repair_warnings": ["QA-README"]}})
    assert cfg.optimization_iterations == 0 and cfg.mode == "repair_only"
    for value in (0, 4, True, "2"):
        with pytest.raises(ValueError):
            QualityLoopConfig.from_dict({"target_gpu_model": "MI355X", "quality_loop": {
                "max_repair_attempts": value}})
    # Blocking checks are always repaired; only warning ids can be opted in.
    for warnings in (["unknown"], ["QA-CONTROLS"], ["QA-README", "QA-README"]):
        with pytest.raises(ValueError, match="QA warning check ids"):
            QualityLoopConfig.from_dict({"target_gpu_model": "MI355X", "quality_loop": {
                "repair_warnings": warnings}})


def test_findings_keep_independent_ids_and_selected_warnings():
    report = qa_report("FAIL", [finding("QA-CONTROLS", "blocking", "zero passes"),
                                finding("QA-README", "warning", "README.md is missing")])
    assert len(blocking_findings(report)) == 1
    assert len(blocking_findings(report, ("QA-README",))) == 2
    previous = blocking_findings(report)
    assert previous == blocking_findings(deepcopy(report))
    report["findings"][0]["detail"] = "another defect"
    assert previous.keys().isdisjoint(blocking_findings(report))


class ChangingBackend:
    def __init__(self):
        self.calls = 0

    def run(self, prompt, workspace, *, role):
        assert role == "repair" and "qa_report.yaml" in prompt and "solution/" in prompt
        self.calls += 1
        edit_cases(workspace, lambda cases: cases.append(dict(EXTRA, id=f"tail{self.calls}",
                                                              axes={"n": 5 + 2 * self.calls})))


def test_repair_only_closes_all_findings_without_optimizer_or_fake_speedup(tmp_path, monkeypatch):
    task = make_task(tmp_path / "worktree")
    backend = ChangingBackend()
    fail = qa_report("FAIL", [finding("QA-COUNTEREXAMPLE", "blocking", "odd tail passes wrongly")])
    loop = StubQualityLoop(tmp_path, QualityLoopConfig.from_dict({
        "target_gpu_model": "MI355X", "quality_loop": {"mode": "repair_only", "max_repair_attempts": 2}}),
        logger=logging.getLogger(__name__), backend=backend, publisher=FakePublisher(),
        reports=[fail, fail, qa_report("PASS")])
    attach_state(loop, tmp_path, task.parents[3])
    monkeypatch.setattr(loop, "_optimize_once", lambda *a: pytest.fail("repair-only optimized"))
    loop._process_task(TASK_ID, task)
    assert backend.calls == 2
    assert [case["id"] for case in json.loads((task / "workloads.json").read_text())["cases"]][-2:] == [
        "tail1", "tail2"]
    record = loop.state.task(TASK_ID)
    assert record["state"] == "completed" and record["speedups"] == []
    assert record["changes"] == ["workloads.json"]
    assert len(record["reviewer"]["resolved_findings"]) == 1


@pytest.mark.parametrize("tamper", [False, True])
def test_repair_review_cannot_omit_findings_or_mutate_original(tmp_path, tamper):
    original = package(tmp_path, "original")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "README.md").write_text("valid source")

    class Reviewer:
        def run(self, prompt, work, *, role):
            assert "task.yaml" in prompt
            if tamper:
                (original / "reference.py").write_text("changed")
            (work / "quality_loop_repair_review.yaml").write_text(yaml.safe_dump(dict(
                accepted=True, contract_preserved=True, scoring_preserved=True,
                resolved_findings=[], summary="unsupported acceptance")))

    loop = QualityLoop(tmp_path, QualityLoopConfig(), logger=logging.getLogger(__name__),
                       reviewer_backend=Reviewer())
    if tamper:
        with pytest.raises(RuntimeError, match="protected files"):
            loop._review_repair(TASK_ID, original, workspace, {"known": {}})
    else:
        assert loop._review_repair(TASK_ID, original, workspace, {"known": {}})["accepted"] is False
