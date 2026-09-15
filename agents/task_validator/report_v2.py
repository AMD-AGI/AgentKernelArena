"""Version-4 validator reports for the schema-v2 task lifecycle.

The report schema is separate from task schema (2) and runner protocol (1).
Only captured framework evidence grants lifecycle skips or diagnostic acceptance.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any, Mapping

from .trusted_evidence import TrustedTaskEvidence, evaluate_task_evidence, snapshot_task_evidence

V2_REPORT_SCHEMA_VERSION = 4
DRAFT_FILENAME = "validation_report.draft.yaml"
SEMANTIC_CHECKS = (
    "correctness_implementation_review", "self_contained", "gpu_hang_check",
    "result_template_compatibility", "benchmark_integrity", "harness_integrity",
)


def compute_v2_overall_status(report: Mapping) -> str:
    from .report_schema import CHECK_NAMES

    if report.get("framework_status") != "PASS" or report.get("initial_validation_gate") != "PASS":
        return "FAIL"
    gating = report.get("baseline_gating", {})
    if not isinstance(gating, dict) or gating.get("accepted") is not True:
        return "FAIL"
    checks = report.get("checks", {})
    if not isinstance(checks, dict) or set(checks) != set(CHECK_NAMES):
        return "FAIL"
    statuses = []
    for name in CHECK_NAMES:
        check = checks[name]
        if not isinstance(check, dict):
            return "FAIL"
        status = check.get("status")
        if name == "correctness" and status == "FAIL":
            if (gating.get("diagnostic_accepted") is True
                    and gating.get("numerical_status") == "FAIL"
                    and gating.get("policy") == "diagnostic"
                    and isinstance(gating.get("diagnostic_reason"), str)
                    and gating["diagnostic_reason"].strip()):
                continue
            return "FAIL"
        if status == "SKIP":
            if (name in ("source_files_exist", "target_symbols_found")
                    and check.get("skip_reason_code") == "candidate_unimplemented"
                    and report.get("candidate_initial_state") == "unimplemented"):
                continue
            return "FAIL"
        if status not in ("PASS", "WARN"):
            return "FAIL"
        statuses.append(status)
    candidate = report.get("candidate_initial_checks")
    if not isinstance(candidate, dict) or set(candidate) != {"compile", "correctness", "performance"}:
        return "FAIL"
    for action, check in candidate.items():
        if not isinstance(check, dict):
            return "FAIL"
        if report.get("candidate_initial_state") == "unimplemented":
            # Performance is not run either; this is a lifecycle absence, never
            # a passed or scoreable candidate measurement.
            if check.get("status") != "SKIP" or check.get("skip_reason_code") != "candidate_unimplemented":
                return "FAIL"
        elif check.get("status") != "PASS":
            # A frozen initial_candidate can share the diagnostic baseline. Its
            # actual FAIL is retained, and this only concerns task validation.
            if not (action == "correctness" and check.get("status") == "FAIL"
                    and check.get("source_role") == "baseline"
                    and gating.get("diagnostic_accepted") is True):
                return "FAIL"
    return "WARN" if "WARN" in statuses else "PASS"


def _action_check(evaluated: dict, role: str, action: str) -> dict:
    result = evaluated.get("results", {}).get((role, action))
    record = evaluated.get("records", {}).get((role, action), {})
    check = {"status": result.status if result else "FAIL", "source_role": role,
             "details": result.reason or "Framework executed the declared action." if result else
             record.get("execution_error", "No completed framework action evidence."),
             "commands": deepcopy(record.get("commands", [])),
             "invocation_id": record.get("invocation_id")}
    if result:
        check["result"] = result.to_mapping()
    return check


def normalize_v2_report(raw_report: Any, *, expected_task_name: str,
                        trusted_task_evidence: Mapping | TrustedTaskEvidence | None,
                        validation_request_id: str | None,
                        framework_error: str | None = None) -> dict:
    from .report_schema import (
        CHECK_NAMES,
        _normalize_benchmark_integrity, _normalize_harness_integrity, _utc_timestamp, _valid_timestamp,
    )

    errors, warnings, findings = [], [], []
    raw = raw_report if isinstance(raw_report, dict) else {}
    if raw.get("validation_schema_version") != V2_REPORT_SCHEMA_VERSION:
        errors.append(f"Model draft requires validation_schema_version: {V2_REPORT_SCHEMA_VERSION}")
    if not validation_request_id or raw.get("validation_request_id") != validation_request_id:
        errors.append("Missing or stale validation_request_id in model draft")
    timestamp = raw.get("validation_timestamp")
    if not _valid_timestamp(timestamp):
        errors.append("Model draft requires a valid validation_timestamp")
        timestamp = _utc_timestamp()
    elif isinstance(timestamp, datetime):
        timestamp = timestamp.isoformat()
    if raw.get("task_name") != expected_task_name:
        errors.append("Model draft task_name does not match the current task")
    evaluated = {}
    context = {}
    digest = None
    try:
        snapshot = snapshot_task_evidence(trusted_task_evidence, task_id=expected_task_name)
        digest = snapshot.sha256
        context = snapshot.to_mapping()
        evaluated = evaluate_task_evidence(snapshot)
        errors.extend(evaluated["errors"])
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        errors.append(f"Invalid trusted task evidence: {exc}")
    if raw.get("task_evidence_sha256") != digest or digest is None:
        errors.append("Model draft is not bound to this framework evidence snapshot")
    if framework_error:
        errors.append(framework_error)
    raw_checks = raw.get("checks", {})
    if not isinstance(raw_checks, dict):
        errors.append("Model checks must be a mapping")
        raw_checks = {}
    if set(raw_checks) - set(CHECK_NAMES):
        errors.append("Model checks contain unknown entries")
    checks = {}
    unimplemented = evaluated.get("candidate_unimplemented", False)
    checks["config_schema"] = {
        "status": "PASS" if evaluated else "FAIL",
        "details": "Framework TaskSpec parsed the captured schema-v2 declaration." if evaluated else
        "No valid captured TaskSpec.",
    }
    for name in ("source_files_exist", "target_symbols_found", *SEMANTIC_CHECKS):
        if unimplemented and name in ("source_files_exist", "target_symbols_found"):
            checks[name] = {"status": "SKIP", "skip_reason_code": "candidate_unimplemented",
                            "scope": "initial_candidate", "details":
                            "Framework confirmed an unimplemented initial candidate; baseline remains checked."}
            continue
        model_check = raw_checks.get(name)
        check = deepcopy(model_check) if isinstance(model_check, dict) else {}
        status = check.get("status")
        allowed = ("PASS", "FAIL") if name in ("source_files_exist", "target_symbols_found") else ("PASS", "FAIL", "WARN")
        if status not in allowed:
            errors.append(f"{name}: a semantic review must report PASS, FAIL, or WARN; no model-authored SKIP")
            status = "FAIL"
        if not isinstance(check.get("details"), str) or not check["details"].strip():
            errors.append(f"{name}: review details are required")
            status = "FAIL"
        evidence = check.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            errors.append(f"{name}: semantic review requires nonempty evidence[] even for PASS")
            status = "FAIL"
        else:
            for item in evidence:
                if (not isinstance(item, dict) or not isinstance(item.get("finding"), str)
                        or not item["finding"].strip() or not any(
                            isinstance(item.get(field), str) and item[field].strip()
                            for field in ("path", "case_id"))):
                    errors.append(f"{name}: each evidence item needs finding and path or case_id")
                    status = "FAIL"
        if name == "correctness_implementation_review" and check.get("is_trivially_passing") is True:
            status = "FAIL"
            findings.append("Correctness review identified a trivially passing checker")
        if name == "benchmark_integrity":
            perf = evaluated.get("results", {}).get(("baseline", "performance"))
            cases = perf.cases if perf else ()
            check["case_count"] = len(cases)
            check["valid_case_count"] = sum(row.get("status") == "PASS" for row in cases)
            check["benchmark_methods"] = sorted({row["benchmark_method"] for row in cases
                                                 if isinstance(row.get("benchmark_method"), str)})
            # Numerical policy is separate from benchmark semantics. All these
            # review fields still come from the model; a command PASS proves
            # neither representative inputs nor fair timing boundaries.
            previous = status
            status = _normalize_benchmark_integrity(check, status, errors, findings)
            if previous == "FAIL":
                status = "FAIL"
        if name == "harness_integrity":
            guard = context.get("harness", {})
            check["framework_guard_enforced"] = guard.get("enforced_during_optimization", False)
            check["protected_paths"] = deepcopy(guard.get("protected_paths", []))
            status = _normalize_harness_integrity(check, status, errors, findings)
        check["status"] = status
        checks[name] = check
    for name, action in (("compilation", "compile"), ("correctness", "correctness"), ("performance", "performance")):
        checks[name] = _action_check(evaluated, "baseline", action)
    spec = evaluated.get("spec")
    candidate = {}
    for action in ("compile", "correctness", "performance"):
        if unimplemented:
            candidate[action] = {"status": "SKIP", "skip_reason_code": "candidate_unimplemented",
                                 "scope": "initial_candidate", "details":
                                 "No initial candidate to measure; final candidate evaluation has no such exception."}
        else:
            role = "baseline" if spec and spec.baseline.kind == "initial_candidate" else "candidate"
            candidate[action] = _action_check(evaluated, role, action)
    report = {
        "validation_schema_version": V2_REPORT_SCHEMA_VERSION,
        "task_schema_version": 2, "validation_phase": "task_validation",
        "validation_request_id": validation_request_id,
        "task_evidence_sha256": digest, "task_name": expected_task_name,
        "validation_timestamp": timestamp, "framework_status": "FAIL" if errors else "PASS",
        "initial_validation_gate": "PASS" if evaluated.get("accepted") else "FAIL",
        "candidate_initial_state": spec.candidate.initial_state if spec else "unknown",
        "baseline_gating": {
            "accepted": evaluated.get("accepted", False),
            "policy": spec.baseline.correctness_policy if spec else None,
            "numerical_status": evaluated.get("baseline_numerical_status", "NOT_RUN"),
            "diagnostic_accepted": evaluated.get("diagnostic", False),
            "diagnostic_reason": spec.baseline.diagnostic_reason if spec else None,
        },
        "checks": {name: checks[name] for name in CHECK_NAMES},
        "candidate_initial_checks": candidate,
        "agent_reported_overall_status": raw.get("overall_status"),
        "validation_errors": errors, "validation_warnings": warnings, "policy_findings": findings,
        "framework_errors": [framework_error] if framework_error else [],
        "summary": raw.get("summary", "") if isinstance(raw.get("summary"), str) else "",
    }
    report["overall_status"] = compute_v2_overall_status(report)
    return report
