"""The production-range exemption from the 100x performance-span rule (workloads.performance_span).

CPU only and light: the pilot rms_norm task is copied into tmp_path with exempt workloads and an
evidence file; the CPU smoke of the pre-check is limited to small cases.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from src.task_contract import MIN_SIZE_SPAN, load_task_contract
from src.task_spec import TaskConfigError, load_task_spec

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "tasks/triton/normalization/rms_norm/bf16_h8192"
PILOT_ID = "triton/normalization/rms_norm/bf16_h8192"

CASES = [
    {"id": "t1", "axes": {"n_tokens": 1}, "checks": ["correctness"], "source": "decode, batch 1"},
    {"id": "t6", "axes": {"n_tokens": 6}, "checks": ["correctness"], "source": "decode"},
    {"id": "t1000", "axes": {"n_tokens": 1000}, "checks": ["correctness"], "source": "unaligned"},
    {"id": "t2048", "axes": {"n_tokens": 2048}, "checks": ["correctness", "performance"], "source": "prefill"},
    {"id": "t4096", "axes": {"n_tokens": 4096}, "checks": ["correctness", "performance"], "source": "prefill"},
    {"id": "t8192", "axes": {"n_tokens": 8192}, "checks": ["correctness", "performance"], "source": "prefill"},
    {"id": "t16384", "axes": {"n_tokens": 16384}, "checks": ["correctness", "performance"], "source": "prefill"},
]
EVIDENCE = {
    "source": "AgentX corpus rev 23f152f6 + model serving config (test fixture)",
    "method": "corpus_statistics",
    "axes": {"n_tokens": {"observed_min": 1, "observed_max": 65536, "perf_min": 2048, "perf_max": 16384,
                          "frequency": {"1": 0.3, "2048": 0.3, "4096": 0.2, "8192": 0.1, "16384": 0.1}}},
    "fixed_dimensions": "hidden_size is fixed by the model",
    "notes": "test fixture",
}


def _workloads(cases=None, span=True, evidence="workload_evidence.json"):
    workloads = {"description": "exempt test workloads", "cases": copy.deepcopy(cases or CASES)}
    if span:
        workloads["performance_span"] = {"policy": "production_range", "evidence": evidence}
    return workloads


def _package(tmp_path: Path, *, workloads=None, evidence=EVIDENCE, name="workload_evidence.json") -> Path:
    root = tmp_path / "tasks" / PILOT_ID
    shutil.copytree(PILOT, root, ignore=shutil.ignore_patterns("__pycache__"))
    (root / "workloads.json").write_text(json.dumps(workloads if workloads is not None else _workloads(), indent=1))
    if evidence is not None:
        (root / name).write_text(json.dumps(evidence, indent=1))
    return root


def _load(root: Path):
    return load_task_contract(root / "task.yaml", task_id=PILOT_ID)


def _evidence(**axis):
    evidence = copy.deepcopy(EVIDENCE)
    evidence["axes"]["n_tokens"].update(axis)
    return evidence


def test_an_exempt_task_with_evidence_loads(tmp_path):
    contract = _load(_package(tmp_path))
    span = contract.performance_span
    assert span["policy"] == "production_range" and span["evidence"] == "workload_evidence.json"
    assert span["perf_span"] == 8.0 and span["observed_span"] == 65536.0  # the observed range may span >= 100x
    assert span["axes"]["n_tokens"] == {"observed_min": 1, "observed_max": 65536, "perf_min": 2048,
                                        "perf_max": 16384, "perf_frequency": pytest.approx(0.7)}
    resolved = contract.resolved_contract("triton")
    assert resolved["performance_span"]["axes"]["n_tokens"]["perf_max"] == 16384
    spec = load_task_spec(_package(tmp_path / "b") / "task.yaml", task_id=PILOT_ID, target_language="hip")
    assert "workload_evidence.json" in spec.baseline.source_files  # part of the read-only task package


def test_the_declaration_requires_its_evidence_file(tmp_path):
    with pytest.raises(TaskConfigError, match="not a file in the task package"):
        _load(_package(tmp_path, evidence=None))
    from src.task_contract import validate_task_contract
    import yaml

    raw = yaml.safe_load((PILOT / "task.yaml").read_text())
    with pytest.raises(TaskConfigError, match="does not contain"):
        validate_task_contract(raw, _workloads(), task_id=PILOT_ID)


def test_a_perf_range_spanning_100x_is_refused(tmp_path):
    cases = CASES[:3] + [
        {"id": f"t{n}", "axes": {"n_tokens": n}, "checks": ["correctness", "performance"]} for n in (128, 1024, 4096, 16384)]
    evidence = _evidence(perf_min=128, frequency={"128": 0.5, "16384": 0.5})
    with pytest.raises(TaskConfigError, match=r"spans 128x .* rule is reachable"):
        _load(_package(tmp_path, workloads=_workloads(cases), evidence=evidence))


def test_perf_cases_must_include_both_endpoints(tmp_path):
    with pytest.raises(TaskConfigError, match="No performance case has n_tokens at perf_max=32768"):
        _load(_package(tmp_path, evidence=_evidence(perf_max=32768, frequency={"2048": 0.5, "32768": 0.5})))
    with pytest.raises(TaskConfigError, match="No performance case has n_tokens at perf_min=1024"):
        _load(_package(tmp_path / "b", evidence=_evidence(perf_min=1024, frequency={"1024": 1.0})))


def test_a_perf_case_outside_the_range_fails(tmp_path):
    with pytest.raises(TaskConfigError, match="t2048: n_tokens=2048 lies outside the evidenced perf range 4096..16384"):
        _load(_package(tmp_path, evidence=_evidence(perf_min=4096, frequency={"4096": 1.0})))


def test_an_undeclared_task_below_100x_still_fails(tmp_path):
    with pytest.raises(TaskConfigError, match=f"must span at least {MIN_SIZE_SPAN}x"):
        _load(_package(tmp_path, workloads=_workloads(span=False), evidence=None))


@pytest.mark.parametrize("evidence, message", [
    (_evidence(perf_min=16384, perf_max=2048), "observed_min <= perf_min <= perf_max <= observed_max"),
    (_evidence(observed_min=4096), "observed_min <= perf_min"),
    (_evidence(observed_max=300000), "inside the axis domain 1..262144"),
    (_evidence(frequency={"2048": 0.2}), "sum to 0.2"),
    (_evidence(frequency={"70000": 1.0}), "not an observed value"),
    (_evidence(frequency={"2048": -1.0, "4096": 2.0}), "nonnegative"),
    (_evidence(perf_max=16384.0), "must be an integer"),
    ({**EVIDENCE, "method": "guess"}, "method must be one of"),
    ({k: v for k, v in EVIDENCE.items() if k != "source"}, "missing fields"),
    ({**EVIDENCE, "axes": {"n_tokens": EVIDENCE["axes"]["n_tokens"], "hidden_size": {}}}, "exactly the var axes"),
    ({**EVIDENCE, "extra": 1}, "unknown fields"),
])
def test_malformed_evidence_is_rejected(tmp_path, evidence, message):
    with pytest.raises(TaskConfigError, match=message):
        _load(_package(tmp_path, evidence=evidence))


@pytest.mark.parametrize("declaration, message", [
    ({"policy": "short_range", "evidence": "workload_evidence.json"}, "policy must be 'production_range'"),
    ({"policy": "production_range"}, "missing fields"),
    ({"policy": "production_range", "evidence": "inputs/evidence.json"}, "at the task root"),
    ({"policy": "production_range", "evidence": "workloads.json"}, "other than workloads.json"),
    ({"policy": "production_range", "evidence": "evidence.yaml"}, ".json file"),
])
def test_malformed_declarations_are_rejected(tmp_path, declaration, message):
    workloads = _workloads()
    workloads["performance_span"] = declaration
    with pytest.raises(TaskConfigError, match=message):
        _load(_package(tmp_path, workloads=workloads))


def test_the_case_count_and_boundary_rules_still_apply(tmp_path):
    with pytest.raises(TaskConfigError, match="4-16 performance cases"):
        _load(_package(tmp_path, workloads=_workloads([c for c in CASES if c["id"] != "t4096"])))
    with pytest.raises(TaskConfigError, match="boundary"):
        _load(_package(tmp_path / "b", workloads=_workloads([c for c in CASES if c["id"] != "t1"])))


def test_existing_tasks_are_unaffected():
    listed = subprocess.run(["git", "ls-files", "tasks"], cwd=ROOT, capture_output=True, text=True)
    if listed.returncode:
        pytest.skip("git is unavailable")
    tasks = [ROOT / path for path in listed.stdout.split() if path.endswith("/task.yaml")]
    assert tasks
    for path in tasks:
        task_id = path.parent.relative_to(ROOT / "tasks").as_posix()
        contract = load_task_contract(path, task_id=task_id)
        if "performance_span" not in contract.workloads:
            assert contract.performance_span is None
            assert "performance_span" not in contract.resolved_contract("triton")


def test_check_task_reports_the_exemption(tmp_path, monkeypatch):
    from src.tools import check_task

    root = _package(tmp_path)
    monkeypatch.setattr(check_task, "ROOT", tmp_path)
    result = check_task.check(root, smoke_limit=1000)  # keeps the CPU smoke to the small cases
    assert result["ok"], result["errors"]
    assert result["span_exemption"]["perf_span"] == 8.0
    assert any("production_range exemption" in warning and "workload_evidence.json" in warning
               for warning in result["warnings"])


def _context(times: dict) -> dict:
    rows = [{"test_case_id": case_id, "status": "PASS", "execution_time_ms": ms,
             "benchmark_method": "cuda_graph_cold_cache"} for case_id, ms in times.items()]
    return {"initial_validation": {"accepted": True, "candidate_checks": "PASS"},
            "actions": [{"result": {"role": "baseline", "action": "performance", "cases": rows}}]}


def test_qa_reports_the_exemption_and_checks_the_endpoints_against_the_floor(tmp_path):
    from agents.qa_auditor.deterministic import check_initial, check_span_exemption
    from agents.qa_auditor.rubric import BLOCKING, CHECKS, WARNING

    assert CHECKS["QA-SPAN-EXEMPTION"].severity == WARNING
    assert CHECKS["QA-SPAN-EVIDENCE"].severity == BLOCKING
    contract = _load(_package(tmp_path))
    good = _context({"t2048": 0.012, "t4096": 0.02, "t8192": 0.04, "t16384": 0.08})
    findings = []
    check_span_exemption(contract.performance_span, contract.workloads, good, findings)
    assert [f.check_id for f in findings] == ["QA-SPAN-EXEMPTION"]
    detail = findings[0].detail
    assert "observed 1..65536, perf 2048..16384" in detail and "AgentX corpus" in detail
    assert findings[0].to_mapping()["severity"] == WARNING

    slow_floor = _context({"t2048": 0.006, "t4096": 0.02, "t8192": 0.04})  # t16384 was never measured
    findings = []
    check_span_exemption(contract.performance_span, contract.workloads, slow_floor, findings)
    evidence = [f for f in findings if f.check_id == "QA-SPAN-EVIDENCE"]
    assert {(f.evidence["case"], f.evidence["endpoint"]) for f in evidence} == {("t2048", "perf_min"),
                                                                                ("t16384", "perf_max")}
    # The exemption does not bypass the 10 us floor for any performance case.
    findings = []
    check_initial(slow_floor, contract.raw, findings)
    assert any(f.check_id == "QA-TIMING-FLOOR" and "t2048" in f.detail for f in findings)


def test_qa_reports_nothing_for_tasks_without_the_exemption():
    from agents.qa_auditor.deterministic import check_span_exemption

    contract = load_task_contract(PILOT / "task.yaml", task_id=PILOT_ID)
    findings = []
    check_span_exemption(contract.performance_span, contract.workloads, _context({"t2048": 0.001}), findings)
    assert findings == []


def test_held_out_cases_of_an_exempt_task_stay_in_the_production_range(tmp_path):
    from src.held_out.run_heldout_eval import heldout_contract

    contract = _load(_package(tmp_path)).resolved_contract("triton")
    inside = [{"id": "h12288", "axes": {"n_tokens": 12288}, "checks": ["correctness", "performance"]},
              {"id": "h40000", "axes": {"n_tokens": 40000}, "checks": ["correctness"]}]
    extended = heldout_contract(contract, salt="s", extra_cases=inside)
    assert [case["id"] for case in extended["cases"][-2:]] == ["h12288", "h40000"]
    with pytest.raises(ValueError, match="outside the task's perf range 2048..16384"):
        heldout_contract(contract, salt="s", extra_cases=[
            {"id": "h1024", "axes": {"n_tokens": 1024}, "checks": ["correctness", "performance"]}])
    with pytest.raises(ValueError, match="outside the task's observed production range 1..65536"):
        heldout_contract(contract, salt="s", extra_cases=[
            {"id": "h100000", "axes": {"n_tokens": 100000}, "checks": ["correctness"]}])
    # A task without the exemption keeps the plain axis-domain rule.
    plain = load_task_contract(PILOT / "task.yaml", task_id=PILOT_ID).resolved_contract("triton")
    heldout_contract(plain, salt="s", extra_cases=[
        {"id": "h100000", "axes": {"n_tokens": 100000}, "checks": ["correctness", "performance"]}])
