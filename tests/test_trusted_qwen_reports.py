"""The phase adapter reuses actual MoE/recurrent/attention report contracts."""
from copy import deepcopy
import json

import pytest

from src.tools import trusted_qwen_eval as adapter, trusted_task_eval as trusted
from test_qwen_moe_quality import moe
from test_qwen_bound_graph_quality import graph_task


def host_receipt(root, contract, phase, report):
    request = {"phase": phase, "request_id": "e" * 48,
               "source_sha256": {name: trusted.sha256(trusted.read_regular(root / name))
                                 for name in contract["sources"]}}
    path = root / "build" / (phase + "_report.json")
    path.write_text(json.dumps(report))
    (root / "build/host_phase_receipt.json").write_text(json.dumps({
        "schema": "trusted-qwen-phase-v1", "host_request": request, "returncode": 0,
        "producer_report_sha256": trusted.sha256(path.read_bytes())}))
    return request


def test_moe_trusted_performance_uses_current_observed_execution(moe):
    root, _, emit = moe
    _, report, archive = emit([1.0] * 100)
    contract = trusted.package_contract(root)
    request = host_receipt(root, contract, "performance", report)
    rows, authority = adapter.validate_phase(root, contract, "performance", report, request)
    assert len(rows) == len(authority) == 2
    archive.unlink()
    with pytest.raises((ValueError, OSError)):
        adapter.validate_phase(root, contract, "performance", report, request)


def test_moe_trusted_correctness_requires_the_original_raw_publication(moe):
    root, _, emit = moe
    raw, _, _ = emit([1.0] * 100)
    contract = trusted.package_contract(root)
    _, helper = adapter.validator(root, contract)
    producer_request = helper["make_request"](root, "correctness", "d" * 48)
    raw.update(request=producer_request, request_sha256=helper["fingerprint"](producer_request))
    for row in raw["cases"]:
        row["validation"]["negative_controls"] = {"no_op": "rejected", "wrong_output": "rejected"}
    raw["effective_execution_sha256"] = helper["effective_execution_identity"](raw, contract["qwen_metadata"], producer_request)
    archive = root / "build/reports" / producer_request["run_id"] / "raw.json"
    archive.parent.mkdir(parents=True); archive.write_text(json.dumps(raw))
    request = host_receipt(root, contract, "correctness", raw)
    assert adapter.validate_phase(root, contract, "correctness", raw, request) == ([], [])
    archive.write_text("{}")
    with pytest.raises(ValueError, match="publication differs"):
        adapter.validate_phase(root, contract, "correctness", raw, request)


def test_bound_graph_trusted_performance_requires_all_original_cases(graph_task):
    _, root, _, emit = graph_task
    raw, report, path = emit([1.0] * 100)
    contract = trusted.package_contract(root)
    request = host_receipt(root, contract, "performance", report)
    rows, authority = adapter.validate_phase(root, contract, "performance", report, request)
    assert len(rows) == len(authority) == contract["qwen_profile"]["cases"]
    raw["cases"].pop(); path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        adapter.validate_phase(root, contract, "performance", report, request)


def test_bound_graph_correctness_requires_original_ledger_and_complete_CPU_truth(graph_task):
    _, root, _, emit = graph_task
    emit([1.0] * 100)
    contract = trusted.package_contract(root)
    _, helper = adapter.validator(root, contract)
    policy = helper["contract"](root)
    raw = {"status": "ok", "mode": "cpu_truth_correctness", **helper["provenance"](root, policy),
           "cases": [{"sig": case["test_case_id"], "params": case["params"], "expected_device": "cpu",
                      "checks": [{"kind": kind, "passed": True} for kind in ("ordinary", *policy["challenges"])],
                      "checked_invocations": 5, "tolerance": policy["tolerance"],
                      "max_rel_err": 0.0, "max_mixed_error_ratio": 0.0} for case in policy["cases"]]}
    # Native launcher provenance is part of attention's correctness contract.
    binding = helper["contract"](root).get("source_binding")
    if binding:
        performance = json.loads((root / "build/_bench_raw.json").read_text())
        raw["source_binding"] = performance["source_binding"]
    (root / "build/_cpu_truth_correctness.json").write_text(json.dumps(raw))
    ledgers = root / "build/ledger-root/reports/ledger"
    ledgers.mkdir(parents=True)
    for case_id in contract["qwen_metadata"]["ledger_ids"]:
        (ledgers / (case_id + ".json")).write_text(json.dumps({"ledger_id": case_id, "status": "PASS"}))
    report = {"status": "ok", "exit_code": 0, "cpu_truth_challenges": deepcopy(raw)}
    request = host_receipt(root, contract, "correctness", report)
    assert adapter.validate_phase(root, contract, "correctness", report, request) == ([], [])
    next(ledgers.iterdir()).unlink()
    with pytest.raises(OSError):
        adapter.validate_phase(root, contract, "correctness", report, request)
