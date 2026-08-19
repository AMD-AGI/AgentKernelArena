# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Deterministic schema and finalization for task-validator reports.

The validation agent is intentionally used for judgment-heavy code review, but
it is not the authority for whether its own report is complete or for computing
the aggregate status.  This module provides that trust boundary.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml


REPORT_SCHEMA_VERSION = 3
REPORT_FILENAME = "validation_report.yaml"
COMPLETION_MARKER_FILENAME = ".validation_complete"

CHECK_NAMES = (
    "config_schema",
    "source_files_exist",
    "target_symbols_found",
    "compilation",
    "correctness",
    "performance",
    "correctness_implementation_review",
    "self_contained",
    "gpu_hang_check",
    "result_template_compatibility",
    "benchmark_integrity",
    "harness_integrity",
)

STATUS_VALUES = frozenset({"PASS", "FAIL", "WARN", "TIMEOUT", "SKIP"})
ALLOWED_STATUSES = {
    "config_schema": frozenset({"PASS", "FAIL", "WARN"}),
    "source_files_exist": frozenset({"PASS", "FAIL", "SKIP"}),
    "target_symbols_found": frozenset({"PASS", "FAIL", "SKIP"}),
    "compilation": frozenset({"PASS", "FAIL", "TIMEOUT", "SKIP"}),
    "correctness": frozenset({"PASS", "FAIL", "TIMEOUT", "SKIP"}),
    "performance": frozenset({"PASS", "FAIL", "WARN", "TIMEOUT", "SKIP"}),
    "correctness_implementation_review": frozenset({"PASS", "FAIL", "WARN", "SKIP"}),
    "self_contained": frozenset({"PASS", "FAIL", "WARN"}),
    "gpu_hang_check": frozenset({"PASS", "FAIL", "WARN"}),
    "result_template_compatibility": frozenset({"PASS", "FAIL", "WARN"}),
    "benchmark_integrity": frozenset({"PASS", "FAIL", "WARN", "SKIP"}),
    "harness_integrity": frozenset({"PASS", "FAIL", "WARN"}),
}

ALLOWED_SKIP_REASONS = frozenset(
    {
        "repository_field_not_declared",
        "generation_placeholder",
        "starter_stub",
        "dependency_failed",
        "not_applicable",
    }
)
COMMAND_CHECKS = frozenset({"compilation", "correctness", "performance"})
SCOREABLE_BENCHMARK_METHODS = frozenset({"cuda_graph", "cuda_event_fallback"})
REVIEW_CHECKS = frozenset(
    {
        "correctness_implementation_review",
        "self_contained",
        "result_template_compatibility",
        "benchmark_integrity",
        "harness_integrity",
    }
)
HARD_BENCHMARK_REVIEW_FIELDS = (
    "method_metadata_complete",
    "method_policy_valid",
    "case_identity_complete",
    "baseline_policy_immutable",
    "state_restore_valid",
    "workload_symmetric",
    "representative_inputs_valid",
    "timing_boundaries_valid",
)
ADVISORY_BENCHMARK_REVIEW_FIELDS = ("replay_validation_valid",)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _valid_timestamp(value: Any) -> bool:
    if isinstance(value, datetime):
        return True
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def _status(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().upper()
    return normalized if normalized in STATUS_VALUES else None


def _details(check: Mapping[str, Any], fallback: str) -> str:
    value = check.get("details") or check.get("analysis")
    return value.strip() if isinstance(value, str) and value.strip() else fallback


def _normalize_attempts(
    check_name: str,
    check: dict[str, Any],
    status: str,
    errors: list[str],
) -> str:
    """Validate command evidence and return its authoritative status."""
    if status == "SKIP":
        return status

    attempts = check.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        errors.append(f"{check_name}: non-SKIP command check requires non-empty attempts[]")
        return "FAIL"

    saw_timeout = False
    saw_failure = False
    for index, attempt in enumerate(attempts):
        if not isinstance(attempt, dict):
            errors.append(f"{check_name}: attempts[{index}] must be a mapping")
            saw_failure = True
            continue
        if not isinstance(attempt.get("command"), str) or not attempt["command"].strip():
            errors.append(f"{check_name}: attempts[{index}].command is required")
            saw_failure = True
        timed_out = attempt.get("timed_out")
        exit_code = attempt.get("exit_code")
        if timed_out is True:
            saw_timeout = True
        elif timed_out is not False:
            errors.append(f"{check_name}: attempts[{index}].timed_out must be boolean")
            saw_failure = True
        if not isinstance(exit_code, int) or isinstance(exit_code, bool):
            errors.append(f"{check_name}: attempts[{index}].exit_code must be an integer")
            saw_failure = True
        elif exit_code != 0:
            saw_failure = True

    # Command exit status is authoritative. A JSON/YAML report can add evidence,
    # but may never turn a nonzero exit or timeout into PASS.
    if saw_timeout:
        return "TIMEOUT"
    if saw_failure:
        return "FAIL"
    if status in {"FAIL", "TIMEOUT"}:
        return status
    return status


def _normalize_benchmark_integrity(
    check: dict[str, Any],
    status: str,
    errors: list[str],
    policy_findings: list[str],
) -> str:
    if status == "SKIP":
        return status

    case_count = check.get("case_count")
    valid_case_count = check.get("valid_case_count")
    methods = check.get("benchmark_methods")
    hard_failure = False
    advisory = False
    if not isinstance(case_count, int) or isinstance(case_count, bool):
        errors.append("benchmark_integrity: case_count must be an integer")
        hard_failure = True
    elif case_count <= 0:
        policy_findings.append(
            "benchmark_integrity: performance produced no scoreable cases"
        )
        hard_failure = True
    if (
        not isinstance(valid_case_count, int)
        or isinstance(valid_case_count, bool)
    ):
        errors.append("benchmark_integrity: valid_case_count must be an integer")
        hard_failure = True
    elif isinstance(case_count, int) and valid_case_count != case_count:
        policy_findings.append(
            "benchmark_integrity: every emitted case must be structurally scoreable"
        )
        hard_failure = True
    if not isinstance(methods, list) or not methods:
        errors.append("benchmark_integrity: benchmark_methods must be a non-empty list")
        hard_failure = True
        normalized_methods: set[str] = set()
    else:
        normalized_methods = {m for m in methods if isinstance(m, str)}
        if not all(isinstance(m, str) for m in methods) or not (
            normalized_methods <= SCOREABLE_BENCHMARK_METHODS
        ):
            policy_findings.append(
                "benchmark_integrity: methods must be exactly cuda_graph or cuda_event_fallback"
            )
            hard_failure = True

    for field in HARD_BENCHMARK_REVIEW_FIELDS:
        if field not in check:
            errors.append(f"benchmark_integrity: missing required field {field}")
            hard_failure = True
            continue
        value = check[field]
        if value is False:
            policy_findings.append(f"benchmark_integrity: {field} is false")
            hard_failure = True
        elif value is None:
            policy_findings.append(f"benchmark_integrity: {field} is undetermined")
            advisory = True
        elif value is not True:
            errors.append(f"benchmark_integrity: {field} must be true, false, or null")
            hard_failure = True

    for field in ADVISORY_BENCHMARK_REVIEW_FIELDS:
        if field not in check:
            errors.append(f"benchmark_integrity: missing required field {field}")
            hard_failure = True
            continue
        value = check[field]
        if value is False:
            policy_findings.append(
                "benchmark_integrity: exact captured-graph replay output is not validated"
            )
            advisory = True
        elif value is None:
            policy_findings.append(f"benchmark_integrity: {field} is undetermined")
            advisory = True
        elif value is not True:
            errors.append(f"benchmark_integrity: {field} must be true, false, or null")
            hard_failure = True

    if "cuda_event_fallback" in normalized_methods:
        reasons = check.get("event_fallback_reasons")
        if not isinstance(reasons, list) or not reasons or not all(
            isinstance(reason, str) and reason.strip() for reason in reasons
        ):
            policy_findings.append(
                "benchmark_integrity: Event fallback requires a non-empty fallback reason"
            )
            hard_failure = True

    if hard_failure:
        return "FAIL"
    if advisory:
        return "WARN"
    return status


def _normalize_harness_integrity(
    check: dict[str, Any],
    status: str,
    errors: list[str],
    policy_findings: list[str],
) -> str:
    invalid = False
    for field in ("guard_coverage_reviewed", "editable_targets_preserved"):
        if field not in check or not isinstance(check[field], bool):
            errors.append(f"harness_integrity: {field} must be boolean")
            invalid = True
        elif check[field] is False:
            policy_findings.append(f"harness_integrity: {field} is false")
            invalid = True
    return "FAIL" if invalid else status


def _review_evidence_warnings(
    check_name: str,
    check: Mapping[str, Any],
    status: str,
    warnings: list[str],
) -> None:
    """Record non-fatal report-quality warnings for judgment-heavy findings."""

    if check_name not in REVIEW_CHECKS or status not in {"FAIL", "WARN"}:
        return
    evidence = check.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        warnings.append(
            f"{check_name}: FAIL/WARN should include a non-empty evidence[] list"
        )
        return
    for index, item in enumerate(evidence):
        if not isinstance(item, Mapping):
            warnings.append(f"{check_name}: evidence[{index}] must be a mapping")
            continue
        finding = item.get("finding")
        if not isinstance(finding, str) or not finding.strip():
            warnings.append(f"{check_name}: evidence[{index}].finding is required")
        path = item.get("path")
        case_id = item.get("case_id")
        if not (isinstance(path, str) and path.strip()) and not (
            isinstance(case_id, str) and case_id.strip()
        ):
            warnings.append(
                f"{check_name}: evidence[{index}] needs a path or case_id"
            )


def compute_overall_status(report: Mapping[str, Any]) -> str:
    if report.get("framework_status") != "PASS":
        return "FAIL"
    checks = report.get("checks")
    if not isinstance(checks, Mapping):
        return "FAIL"
    statuses = [_status(checks.get(name, {}).get("status")) for name in CHECK_NAMES]
    if any(status in {None, "FAIL", "TIMEOUT"} for status in statuses):
        return "FAIL"
    if any(status == "WARN" for status in statuses):
        return "WARN"
    return "PASS"


def normalize_report(
    raw_report: Any,
    *,
    expected_task_name: str,
    framework_error: str | None = None,
) -> dict[str, Any]:
    """Normalize untrusted agent YAML into a complete versioned report."""
    errors: list[str] = []
    warnings: list[str] = []
    policy_findings: list[str] = []
    raw = raw_report if isinstance(raw_report, dict) else {}
    if not isinstance(raw_report, dict):
        errors.append("validation_report.yaml must contain a YAML mapping")
    if raw.get("validation_schema_version") != REPORT_SCHEMA_VERSION:
        errors.append(
            f"validation_schema_version must be {REPORT_SCHEMA_VERSION}"
        )

    reported_task_name = raw.get("task_name")
    if reported_task_name != expected_task_name:
        # The framework already knows the authoritative task from the workspace
        # it materialized.  A model typo in this display field is report-quality
        # noise, not evidence that the task itself is invalid.
        warnings.append(
            f"task_name mismatch: expected {expected_task_name!r}, got {reported_task_name!r}"
        )

    timestamp = raw.get("validation_timestamp")
    if not _valid_timestamp(timestamp):
        errors.append("validation_timestamp must be a valid ISO 8601 timestamp")
        timestamp = _utc_timestamp()
    elif isinstance(timestamp, datetime):
        timestamp = timestamp.isoformat()

    reported_overall = _status(raw.get("overall_status"))
    if reported_overall not in {"PASS", "FAIL", "WARN"}:
        reported_overall = None

    raw_checks = raw.get("checks")
    if not isinstance(raw_checks, dict):
        errors.append("checks must be a mapping")
        raw_checks = {}
    else:
        unexpected_checks = sorted(set(raw_checks) - set(CHECK_NAMES))
        if unexpected_checks:
            errors.append(f"checks contains unexpected entries: {unexpected_checks}")

    normalized_checks: dict[str, dict[str, Any]] = {}
    skip_reason_sources = {
        "target_symbols_found": ("source_files_exist",),
        "compilation": ("target_symbols_found", "source_files_exist"),
        "correctness": ("compilation", "target_symbols_found"),
        "performance": ("correctness", "compilation"),
        "correctness_implementation_review": (
            "correctness",
            "performance",
            "target_symbols_found",
        ),
        "benchmark_integrity": ("performance", "correctness"),
    }
    for check_name in CHECK_NAMES:
        raw_check = raw_checks.get(check_name)
        if not isinstance(raw_check, dict):
            errors.append(f"{check_name}: missing check mapping")
            normalized_checks[check_name] = {
                "status": "FAIL",
                "details": "Required validator check was missing or malformed.",
            }
            continue

        check = dict(raw_check)
        status = _status(check.get("status"))
        if status not in ALLOWED_STATUSES[check_name]:
            errors.append(
                f"{check_name}: invalid status {check.get('status')!r}; "
                f"allowed={sorted(ALLOWED_STATUSES[check_name])}"
            )
            status = "FAIL"

        if status == "SKIP":
            reason = check.get("skip_reason_code")
            if reason not in ALLOWED_SKIP_REASONS:
                inherited_reason = next(
                    (
                        normalized_checks[source].get("skip_reason_code")
                        for source in skip_reason_sources.get(check_name, ())
                        if source in normalized_checks
                        and normalized_checks[source].get("status") == "SKIP"
                        and normalized_checks[source].get("skip_reason_code")
                        in ALLOWED_SKIP_REASONS
                    ),
                    None,
                )
                if inherited_reason is None:
                    errors.append(
                        f"{check_name}: SKIP requires an allowlisted skip_reason_code"
                    )
                    status = "FAIL"
                else:
                    check["skip_reason_code"] = inherited_reason
                    warnings.append(
                        f"{check_name}: inherited SKIP reason {inherited_reason!r} "
                        "from an upstream check"
                    )

        if check_name in COMMAND_CHECKS:
            status = _normalize_attempts(check_name, check, status, errors)
        elif check_name == "benchmark_integrity":
            status = _normalize_benchmark_integrity(
                check, status, errors, policy_findings
            )
        elif check_name == "harness_integrity":
            status = _normalize_harness_integrity(
                check, status, errors, policy_findings
            )

        check["status"] = status
        check["details"] = _details(check, "No details were supplied by the validation agent.")
        _review_evidence_warnings(check_name, check, status, warnings)
        normalized_checks[check_name] = check

    if framework_error:
        errors.append(framework_error)

    # Cross-check dependency handling. A missing/failed prerequisite cannot be
    # hidden by claiming that a downstream command passed.
    compilation_status = normalized_checks["compilation"]["status"]
    correctness_status = normalized_checks["correctness"]["status"]
    performance_status = normalized_checks["performance"]["status"]
    benchmark_status = normalized_checks["benchmark_integrity"]["status"]
    if compilation_status in {"FAIL", "TIMEOUT"} and correctness_status in {"PASS", "WARN"}:
        errors.append("correctness cannot pass after compilation failed or timed out")
        normalized_checks["correctness"]["status"] = "FAIL"
    if normalized_checks["correctness"]["status"] in {"FAIL", "TIMEOUT"} and performance_status in {
        "PASS",
        "WARN",
    }:
        errors.append("performance cannot pass after correctness failed or timed out")
        normalized_checks["performance"]["status"] = "FAIL"
    if normalized_checks["performance"]["status"] == "SKIP" and benchmark_status in {
        "PASS",
        "WARN",
    }:
        skip_reason = normalized_checks["performance"].get("skip_reason_code")
        normalized_checks["benchmark_integrity"]["status"] = "SKIP"
        normalized_checks["benchmark_integrity"]["skip_reason_code"] = skip_reason
        warnings.append(
            "benchmark_integrity: normalized to SKIP because performance was "
            f"SKIP/{skip_reason}"
        )
    elif normalized_checks["performance"]["status"] in {
        "FAIL",
        "TIMEOUT",
    } and benchmark_status in {"PASS", "WARN"}:
        errors.append("benchmark_integrity cannot pass when performance is not runnable")
        normalized_checks["benchmark_integrity"]["status"] = "FAIL"

    report: dict[str, Any] = {
        "validation_schema_version": REPORT_SCHEMA_VERSION,
        "task_name": expected_task_name,
        "validation_timestamp": timestamp,
        "framework_status": "FAIL" if framework_error or errors else "PASS",
        "agent_reported_overall_status": reported_overall,
        "overall_status": "FAIL",  # Recomputed below.
        "checks": normalized_checks,
        "validation_errors": errors,
        "validation_warnings": warnings,
        "policy_findings": policy_findings,
        "summary": raw.get("summary", "") if isinstance(raw.get("summary"), str) else "",
    }
    report["overall_status"] = compute_overall_status(report)
    return report


def _atomic_yaml_dump(path: Path, data: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w") as handle:
        yaml.safe_dump(dict(data), handle, default_flow_style=False, sort_keys=False)
    temporary.replace(path)


def finalize_report(
    workspace: str | Path,
    *,
    expected_task_name: str,
    framework_error: str | None = None,
) -> dict[str, Any]:
    """Normalize the agent report and atomically mark it framework-complete."""
    workspace_path = Path(workspace)
    report_path = workspace_path / REPORT_FILENAME
    marker_path = workspace_path / COMPLETION_MARKER_FILENAME
    marker_path.unlink(missing_ok=True)

    raw_report: Any = None
    if report_path.exists():
        try:
            with report_path.open() as handle:
                raw_report = yaml.safe_load(handle)
        except Exception as exc:  # Malformed agent output is data, not a launcher crash.
            framework_error = framework_error or f"Unable to parse agent report: {exc}"

    if raw_report is None and framework_error is None:
        framework_error = "Validator backend did not produce validation_report.yaml"

    if isinstance(raw_report, dict):
        # Harness coverage is enforced by framework code outside the task
        # workspace. Replace the agent's guess with the actual effective guard
        # boundary before normalizing the report.
        try:
            from src.harness_guard import describe_workspace_harness

            guard_facts = describe_workspace_harness(workspace_path)
            checks = raw_report.get("checks")
            if isinstance(checks, dict):
                harness_check = checks.get("harness_integrity")
                if isinstance(harness_check, dict):
                    harness_check["framework_guard_enforced"] = bool(
                        guard_facts["enforced_during_optimization"]
                    )
                    harness_check["guard_coverage_reviewed"] = True
                    harness_check["protected_paths"] = guard_facts["protected_paths"]
        except Exception as exc:
            framework_error = framework_error or (
                f"Unable to resolve framework harness facts: {exc}"
            )

    report = normalize_report(
        raw_report,
        expected_task_name=expected_task_name,
        framework_error=framework_error,
    )
    _atomic_yaml_dump(report_path, report)
    digest = hashlib.sha256(report_path.read_bytes()).hexdigest()
    _atomic_yaml_dump(
        marker_path,
        {"validation_schema_version": REPORT_SCHEMA_VERSION, "report_sha256": digest},
    )
    return report


def validation_report_is_complete(workspace: str | Path) -> bool:
    """Return whether a validator report was finalized by the framework."""
    workspace_path = Path(workspace)
    report_path = workspace_path / REPORT_FILENAME
    marker_path = workspace_path / COMPLETION_MARKER_FILENAME
    if not report_path.is_file() or not marker_path.is_file():
        return False
    try:
        with marker_path.open() as handle:
            marker = yaml.safe_load(handle)
        with report_path.open() as handle:
            report = yaml.safe_load(handle)
    except Exception:
        return False
    if not isinstance(marker, dict) or not isinstance(report, dict):
        return False
    if marker.get("validation_schema_version") != REPORT_SCHEMA_VERSION:
        return False
    if marker.get("report_sha256") != hashlib.sha256(report_path.read_bytes()).hexdigest():
        return False
    if report.get("validation_schema_version") != REPORT_SCHEMA_VERSION:
        return False
    checks = report.get("checks")
    if not isinstance(checks, dict) or set(checks) != set(CHECK_NAMES):
        return False
    if any(
        not isinstance(checks[name], dict)
        or _status(checks[name].get("status")) not in ALLOWED_STATUSES[name]
        for name in CHECK_NAMES
    ):
        return False
    return report.get("overall_status") == compute_overall_status(report)
