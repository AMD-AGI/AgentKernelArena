"""QA auditor: finalized reports, the PASS rule, deterministic package checks and counterexample domains.

CPU only. The pilot v3 task is copied and rendered into tmp_path; the GPU control
suite and the candidate evaluator subprocess are replaced where a test reaches them.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil

import pytest
import yaml

from agents.qa_auditor import deterministic, review
from agents.qa_auditor.deterministic import Finding, check_package, check_provenance, check_reference
from agents.qa_auditor.report import (
    COMPLETION_MARKER_FILENAME as MARKER, REPORT_FILENAME as REPORT, build_report, load_report,
    qa_report_is_complete, report_is_current, write_report,
)
from agents.qa_auditor.rubric import BLOCKING, CHECKS, RUBRIC_VERSION, WARNING
from src.task_contract import case_seeds, load_task_contract, render_workspace
from src.task_spec import TaskConfigError

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "tasks/triton/normalization/rms_norm/bf16_h8192"
PILOT_ID = "triton/normalization/rms_norm/bf16_h8192"
SHA = "a" * 64
BLOCKING_IDS = sorted(i for i, c in CHECKS.items() if c.severity == BLOCKING)
WARNING_IDS = sorted(i for i, c in CHECKS.items() if c.severity == WARNING)
REFERENCE, KERNEL = (PILOT / "reference.py").read_text(), (PILOT / "baseline/kernel.py").read_text()
README, TASK_YAML = (PILOT / "README.md").read_text(), (PILOT / "task.yaml").read_text()
WRONG = "def run(*, x, weight, epsilon, out):\n    out.zero_()\n"


def finding(check_id: str, detail: str = "detail") -> dict:
    return Finding(check_id, detail).to_mapping()


def report_for(findings=(), **overrides) -> dict:
    arguments = dict(task_id=PILOT_ID, findings=findings, task_package_sha256=SHA, target_language="triton",
                     request_id="request-1")
    return build_report(**{**arguments, **overrides})


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "ws"
    shutil.copytree(PILOT, ws, ignore=shutil.ignore_patterns("__pycache__"))
    render_workspace(PILOT, ws, task_id=PILOT_ID, target_language="triton")
    return ws


def edit_task(ws: Path, mutate) -> None:
    raw = yaml.safe_load((ws / "task.yaml").read_text())
    mutate(raw)
    (ws / "task.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))


def raw_contract(ws: Path) -> dict:
    return load_task_contract(ws / "task.yaml", task_id=PILOT_ID).raw


# --- report finalization -------------------------------------------------------------------------

def test_written_report_round_trips_with_its_completion_marker(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    report = report_for([finding("QA-README")])
    write_report(tmp_path, report, state=state)
    data = (tmp_path / REPORT).read_bytes()
    assert json.loads((tmp_path / MARKER).read_text()) == {"report_sha256": hashlib.sha256(data).hexdigest(),
                                                           "status": "PASS"}
    assert load_report(tmp_path) == report and qa_report_is_complete(tmp_path)
    assert report_is_current(load_report(tmp_path), SHA) and report["rubric_version"] == RUBRIC_VERSION
    assert (state / REPORT).read_bytes() == data and not list(tmp_path.glob(".*.tmp"))


def _edit(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    assert old in text
    path.write_text(text.replace(old, new))


def _reseal(ws: Path, mutate) -> None:
    """Edit the report and recompute a matching marker: only the consistency rules can reject it."""
    report = yaml.safe_load((ws / REPORT).read_text())
    mutate(report)
    text = yaml.safe_dump(report, sort_keys=False)
    (ws / REPORT).write_text(text)
    (ws / MARKER).write_text(json.dumps({"report_sha256": hashlib.sha256(text.encode()).hexdigest(),
                                         "status": report["status"]}))


def _stale(ws: Path) -> None:
    earlier = (ws / MARKER).read_text()
    write_report(ws, report_for([finding("QA-CONTROLS")], request_id="request-2"))
    (ws / MARKER).write_text(earlier)


def _symlink(ws: Path) -> None:
    (ws / REPORT).rename(ws / "elsewhere.yaml")
    (ws / REPORT).symlink_to(ws / "elsewhere.yaml")


TAMPERING = {
    "none": lambda ws: None,
    "resealed_unchanged": lambda ws: _reseal(ws, lambda r: None),
    "report_status": lambda ws: _edit(ws / REPORT, "status: FAIL", "status: PASS"),
    "report_findings": lambda ws: _edit(ws / REPORT, "zero_output control passes", "nothing"),
    "marker_status": lambda ws: _edit(ws / MARKER, '"status": "FAIL"', '"status": "PASS"'),
    "marker_digest": lambda ws: (ws / MARKER).write_text(json.dumps({"report_sha256": SHA, "status": "FAIL"})),
    "marker_missing": lambda ws: (ws / MARKER).unlink(),
    "marker_not_json": lambda ws: (ws / MARKER).write_text("FAIL\n"),
    "stale_marker": _stale,
    "report_symlink": _symlink,
    "resealed_pass_with_blocker": lambda ws: _reseal(ws, lambda r: r.update(status="PASS")),
    "resealed_blocking_count": lambda ws: _reseal(ws, lambda r: r.update(blocking=0)),
    "resealed_report_version": lambda ws: _reseal(ws, lambda r: r.update(qa_report_version=2)),
}


@pytest.mark.parametrize("tampering", sorted(TAMPERING))
def test_altered_missing_or_stale_report_is_not_complete(tmp_path, tampering):
    write_report(tmp_path, report_for([finding("QA-CONTROLS", "the zero_output control passes correctness"),
                                       finding("QA-README")]))
    TAMPERING[tampering](tmp_path)
    accepted = tampering in ("none", "resealed_unchanged")
    assert (load_report(tmp_path) is not None) is accepted
    assert qa_report_is_complete(tmp_path) is accepted


def test_report_is_current_only_for_the_audited_package_and_rubric(monkeypatch):
    report = report_for()
    assert report_is_current(report, SHA)
    assert not report_is_current(report, "b" * 64)
    assert not report_is_current(report_for(task_package_sha256=None), SHA)
    assert not report_is_current({**report, "rubric_version": "2026-01-01.0"}, SHA)
    monkeypatch.setattr("agents.qa_auditor.report.RUBRIC_VERSION", "a-later-rubric")
    assert not report_is_current(report, SHA)


# --- the PASS rule -------------------------------------------------------------------------------

@pytest.mark.parametrize("ids, status", [
    ([], "PASS"),
    (WARNING_IDS, "PASS"),
    *[([check_id], "FAIL") for check_id in BLOCKING_IDS],
    (WARNING_IDS + ["QA-COUNTEREXAMPLE"], "FAIL"),
])
def test_status_is_pass_exactly_when_no_finding_blocks(tmp_path, ids, status):
    report = report_for([finding(check_id) for check_id in ids])
    blocking = [i for i in ids if CHECKS[i].severity == BLOCKING]
    assert report["status"] == status
    assert (report["blocking"], report["warnings"]) == (len(blocking), len(ids) - len(blocking))
    assert report["checks"] == {i: "FAIL" if i in blocking else "WARN" if i in ids else "PASS" for i in CHECKS}
    severities = [item["severity"] for item in report["findings"]]
    assert severities == sorted(severities, key=lambda severity: severity != BLOCKING)
    write_report(tmp_path, report)
    assert load_report(tmp_path)["status"] == status


def test_framework_errors_block_and_unknown_findings_are_refused():
    report = report_for([finding("QA-README")], framework_errors=["", "launcher crashed"])
    assert report["status"] == "FAIL" and [f["id"] for f in report["findings"]] == ["QA-AUDIT", "QA-README"]
    assert report["findings"][0]["detail"] == "launcher crashed"
    for bad in ({**finding("QA-README"), "id": "QA-INVENTED"}, {**finding("QA-README"), "severity": "error"}):
        with pytest.raises(ValueError, match="Invalid finding"):
            report_for([bad])


# --- deterministic checks on a rendered v3 workspace ---------------------------------------------

def test_clean_pilot_has_no_package_reference_or_provenance_finding(workspace):
    findings: list = []
    raw = raw_contract(workspace)
    check_package(workspace, findings)
    check_reference(workspace, raw, findings)
    check_provenance(workspace, raw, findings)
    assert findings == []


@pytest.mark.parametrize("path, text, expected", [
    ("reference.py", "from baseline.kernel import run\n" + REFERENCE, {"QA-REFERENCE": "imports baseline.kernel"}),
    ("reference.py", "import aiter\n" + REFERENCE, {"QA-PACKAGE": "imports aiter", "QA-REFERENCE": "aiter"}),
    ("reference.py", "from vllm import _custom_ops\n" + REFERENCE,
     {"QA-PACKAGE": "imports vllm", "QA-REFERENCE": "imports vllm"}),
    ("reference.py", "import triton\n" + REFERENCE, {"QA-REFERENCE": "imports triton"}),
    ("reference.py", REFERENCE + "\n\ndef _add():\n    return torch.ops.aten.add\n",
     {"QA-PACKAGE": "uses torch.ops", "QA-REFERENCE": "imports torch.ops"}),
    ("baseline/kernel.py", KERNEL + "\nfrom aiter.ops.triton import rmsnorm\n",
     {"QA-PACKAGE": "imports aiter.ops.triton; vendor"}),
    ("baseline/kernel.py", "import subprocess\n" + KERNEL, {"QA-PACKAGE": "uses subprocess"}),
    ("baseline/kernel.py", KERNEL + '\nLIBRARY = "/opt/rocm/lib"\n', {"QA-PACKAGE": "absolute path /opt/rocm/lib"}),
    ("baseline/kernel.py", KERNEL + "# ported from C:\\Users\\carol\\vllm\n", {"QA-PACKAGE": "local filesystem path"}),
    ("README.md", README + "\nPorted from /home/alice/src/vllm/batch_invariant.py\n",
     {"QA-PACKAGE": "README.md names a local filesystem path '/home/alice/src/vllm/batch_invariant.py'"}),
    ("task.yaml", TASK_YAML + "\n# copied from /Users/bob/vllm\n", {"QA-PACKAGE": "'/Users/bob/vllm'"}),
    ("baseline/kernel.hip", "#include <rocblas/rocblas.h>\n", {"QA-PACKAGE": "includes rocblas/rocblas.h"}),
    ("baseline/kernel.so", "\x7fELF", {"QA-PACKAGE": "is a binary"}),
    ("baseline/._kernel.py", "\x00\x05\x16\x07Mac OS X        \x00\x02", {"QA-PACKAGE": "operating-system metadata"}),
    # The framework-generated candidate copy is not part of the task package.
    ("solution/kernel.py", KERNEL + "import subprocess\n# /home/alice/scratch\n", {}),
])
def test_package_and_reference_violations_are_blocking(workspace, path, text, expected):
    (workspace / path).write_text(text)
    findings: list = []
    raw = raw_contract(workspace)
    check_package(workspace, findings)
    check_reference(workspace, raw, findings)
    check_provenance(workspace, raw, findings)
    details: dict = {}
    for item in findings:
        details.setdefault(item.check_id, []).append(item.detail)
    assert set(details) == set(expected)
    for check_id, needle in expected.items():
        assert any(needle in detail for detail in details[check_id]), details[check_id]
    assert all(item.to_mapping()["severity"] == BLOCKING for item in findings)


@pytest.mark.parametrize("update, license_file, expected", [
    ({"revision": "main"}, False, ["revision 'main' is not a fixed commit"]),
    ({"revision": "HEAD"}, False, ["revision 'HEAD' is not a fixed commit"]),
    ({"revision": " master "}, False, ["revision 'master' is not a fixed commit"]),
    ({"source": "TODO"}, False, ["provenance.source is a placeholder"]),
    ({"license": "unknown"}, False, ["provenance.license is a placeholder", "unknown baseline code needs"]),
    ({"license": "MIT"}, False, ["MIT baseline code needs baseline/LICENSE"]),
    ({"license": "BSD-3-Clause"}, False, ["BSD-3-Clause baseline code needs baseline/LICENSE"]),
    ({"license": "MIT"}, True, []),
    ({"license": "task-original"}, False, []),
    ({"revision": "3f1c2a9d8e7b"}, False, []),
])
def test_provenance_needs_a_fixed_revision_and_a_license_file(workspace, update, license_file, expected):
    edit_task(workspace, lambda raw: raw["baseline"]["provenance"].update(update))
    if license_file:
        (workspace / "baseline/LICENSE").write_text("MIT License\n")
    findings: list = []
    check_provenance(workspace, raw_contract(workspace), findings)
    assert [item.check_id for item in findings] == ["QA-PROVENANCE"] * len(expected)
    assert all(needle in item.detail for item, needle in zip(findings, expected))


def _context(**initial) -> dict:
    case = {"test_case_id": "t2048", "benchmark_method": "cuda_graph_cold_cache", "execution_time_ms": 0.05}
    case.update(initial.pop("case", {}))
    return {"task_id": PILOT_ID, "task_config": {"contract": {"baseline_language": "triton"}},
            "initial_validation": {"accepted": True, "candidate_checks": "PASS", **initial},
            "actions": [{"result": {"role": "baseline", "action": "performance", "cases": [case]}}]}


@pytest.mark.parametrize("arch, initial, expected", [
    ("gfx950", {}, []),
    ("gfx942", {}, ["QA-UNVERIFIED-ARCH"]),
    ("gfx950", {"candidate_checks": "FAIL"}, ["QA-POSITIVE"]),
    ("gfx950", {"case": {"execution_time_ms": 0.004}}, ["QA-TIMING-FLOOR"]),
    ("gfx950", {"case": {"benchmark_method": "hip_events"}}, ["QA-TIMING-METHOD"]),
    ("gfx950", {"accepted": False, "errors": ["baseline compile failed"]}, ["QA-INITIAL"]),
])
def test_run_deterministic_on_the_pilot(workspace, tmp_path, monkeypatch, arch, initial, expected):
    monkeypatch.setenv("PYTORCH_ROCM_ARCH", arch)
    ran = []
    monkeypatch.setattr(deterministic, "run_controls", lambda ws, *rest: ran.append(ws) or {"status": "COMPLETE"})
    context = _context(**initial)
    findings, evidence = deterministic.run_deterministic(workspace, context, tmp_path, controls_timeout_s=1)
    assert [item.check_id for item in findings] == expected
    accepted = context["initial_validation"]["accepted"]
    assert ran == ([workspace] if accepted else []) and evidence == ({"controls": {"status": "COMPLETE"}}
                                                                     if accepted else {})
    report = report_for([item.to_mapping() for item in findings])
    assert report["status"] == ("FAIL" if any(CHECKS[i].severity == BLOCKING for i in expected) else "PASS")


def test_missing_revision_is_a_contract_failure(workspace, tmp_path):
    edit_task(workspace, lambda raw: raw["baseline"]["provenance"].pop("revision"))
    with pytest.raises(TaskConfigError, match="revision"):
        raw_contract(workspace)
    findings, _ = deterministic.run_deterministic(workspace, _context(accepted=False), tmp_path, controls_timeout_s=1)
    assert [item.check_id for item in findings] == ["QA-SPEC", "QA-INITIAL"] and "revision" in findings[0].detail


# --- counterexample case domain ------------------------------------------------------------------

@pytest.fixture
def stepped(workspace):
    """The pilot's resolved contract with n_tokens restricted to multiples of 16 in [16, 4096]."""
    contract = json.loads((workspace / "_aka_eval/contract.json").read_text())
    contract["axes"]["n_tokens"].update(min=16, max=4096, multiple_of=16)
    return contract


@pytest.mark.parametrize("case", [
    {"axes": {"n_tokens": 0}}, {"axes": {"n_tokens": 8}}, {"axes": {"n_tokens": 4112}},
    {"axes": {"n_tokens": 40}}, {"axes": {"n_tokens": 32.0}}, {"axes": {"n_tokens": "32"}},
    {"axes": {"n_tokens": True}}, {"axes": {}}, {"scalars": {"epsilon": 1e-5}},
    {"axes": {"n_tokens": 32, "hidden_size": 8192}}, {"axes": {"n_tokens": 32, "batch": 2}},
    {"axes": {"n_tokens": 32}, "scalars": {"weight": 1.0}}, {"axes": {"n_tokens": 32}, "scalars": {"gamma": 1.0}},
    {"axes": {"n_tokens": 32}, "seeds": [1, 2]}, {"axes": [("n_tokens", 32)]}, [("axes", {})], None,
])
def test_out_of_domain_counterexample_case_is_rejected(stepped, case):
    with pytest.raises(ValueError, match="counterexample"):
        review._validated_case(stepped, copy.deepcopy(case))


@pytest.mark.parametrize("case", [
    {"axes": {"n_tokens": 16}}, {"axes": {"n_tokens": 4096}, "scalars": {"epsilon": 1e-5}},
    {"axes": {"n_tokens": 48}, "scalars": {}},
])
def test_in_domain_counterexample_case_is_accepted(stepped, case):
    assert review._validated_case(stepped, case) == {"axes": case["axes"], "scalars": case.get("scalars", {})}


@pytest.mark.parametrize("counterexample, reason", [
    (None, "no counterexample solution"),
    ({"case": {"axes": {"n_tokens": 5}}}, "no counterexample solution"),
    ({"solution": {"kernel.py": WRONG}, "case": {"axes": {"n_tokens": 0}}}, "outside the declared domain"),
    ({"solution": {"kernel.py": WRONG}, "case": {"axes": {"n_tokens": 262145}}}, "outside the declared domain"),
    ({"solution": {"kernel.py": WRONG}, "case": {"axes": {"n_tokens": 5, "hidden_size": 8192}}}, "must name exactly"),
    ({"solution": {"kernel.py": WRONG}, "case": {"axes": {"n_tokens": 5}, "scalars": {"x": 1}}}, "scalar inputs"),
    ({"solution": {"../kernel.py": WRONG}, "case": {"axes": {"n_tokens": 5}}}, "flat names"),
    ({"solution": {".pth": WRONG}, "case": {"axes": {"n_tokens": 5}}}, "flat names"),
    ({"solution": {"kernel.py": None}, "case": {"axes": {"n_tokens": 5}}}, "flat names"),
])
def test_invalid_counterexample_is_rejected_before_any_evaluation(workspace, tmp_path, monkeypatch,
                                                                  counterexample, reason):
    monkeypatch.setattr(review, "_evaluate", lambda *args: pytest.fail("the evaluator was launched"))
    reproduced, why = review.verify_counterexample(workspace, tmp_path / "scratch", counterexample, 1)
    assert not reproduced and reason in why and not (tmp_path / "scratch").exists()


@pytest.mark.parametrize("task_accepts, also_failed, reproduced, reason", [
    (True, [], True, "reproduced twice"),
    (False, [], False, "the task rejects the counterexample: t1 exceeds the bound"),
    (True, ["t1"], False, "did not isolate a failure (failed: ['qa_counterexample', 't1'])"),
])
def test_counterexample_must_pass_the_task_and_fail_only_the_added_case_twice(
        workspace, tmp_path, monkeypatch, task_accepts, also_failed, reproduced, reason):
    contract_before = (workspace / "_aka_eval/contract.json").read_text()
    added = {"id": review.COUNTEREXAMPLE_CASE_ID, "axes": {"n_tokens": 5}, "scalars": {}, "checks": ["correctness"],
             "seeds": case_seeds(PILOT_ID, review.COUNTEREXAMPLE_CASE_ID)}
    seen = []

    def evaluate(root, timeout_s):
        assert root.parent == tmp_path / "scratch" and [p.name for p in (root / "solution").iterdir()] == ["kernel.py"]
        assert (root / "solution/kernel.py").read_text() == WRONG
        cases = json.loads((root / "_aka_eval/contract.json").read_text())["cases"]
        seen.append([case for case in cases if case["id"] == added["id"]])
        if not seen[-1]:
            return {"status": "PASS" if task_accepts else "FAIL", "reason": "t1 exceeds the bound"}
        rows = [{"test_case_id": c, "status": "FAIL"} for c in [added["id"], *also_failed]]
        return {"status": "FAIL", "cases": rows + [{"test_case_id": "t3", "status": "PASS"}]}

    monkeypatch.setattr(review, "_evaluate", evaluate)
    result = review.verify_counterexample(
        workspace, tmp_path / "scratch", {"solution": {"kernel.py": WRONG}, "case": {"axes": {"n_tokens": 5}}}, 1)
    assert result[0] is reproduced and reason in result[1]
    assert seen == ([[], [added]] * 2 if reproduced else [[], [added]] if task_accepts else [[]])
    assert (workspace / "_aka_eval/contract.json").read_text() == contract_before
    assert (workspace / "solution/kernel.py").read_text() == KERNEL
