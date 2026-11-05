# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""The framework-finalized QA report: itemized findings, PASS only without blockers."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import uuid
from typing import Any, Iterable

import yaml

from .rubric import BLOCKING, CHECKS, RUBRIC_VERSION, WARNING

REPORT_VERSION = 1
REPORT_FILENAME = "qa_report.yaml"
COMPLETION_MARKER_FILENAME = ".qa_complete"
RESERVED_FILENAMES = (REPORT_FILENAME, COMPLETION_MARKER_FILENAME)


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_report(*, task_id: str, findings: Iterable[dict], task_package_sha256: str | None,
                 target_language: str | None, request_id: str | None, evidence: dict | None = None,
                 framework_errors: Iterable[str] = ()) -> dict:
    items = [dict(item) for item in findings]
    for error in framework_errors:
        if error:
            items.append({"id": "QA-AUDIT", "severity": BLOCKING, "title": CHECKS["QA-AUDIT"].title,
                          "detail": str(error)[:4000], "evidence": {}, "fix": CHECKS["QA-AUDIT"].fix})
    for item in items:
        if item.get("severity") not in (BLOCKING, WARNING) or item.get("id") not in CHECKS:
            raise ValueError(f"Invalid finding {item!r}")
    items.sort(key=lambda item: (item["severity"] != BLOCKING, item["id"]))
    blocking = sum(item["severity"] == BLOCKING for item in items)
    checks = {}
    for check_id, definition in CHECKS.items():
        own = [item for item in items if item["id"] == check_id]
        checks[check_id] = ("FAIL" if any(i["severity"] == BLOCKING for i in own) else
                            "WARN" if own else "PASS")
    return {
        "qa_report_version": REPORT_VERSION,
        "rubric_version": RUBRIC_VERSION,
        "task_id": task_id,
        "task_package_sha256": task_package_sha256,
        "target_language": target_language,
        "audit_request_id": request_id or uuid.uuid4().hex,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "FAIL" if blocking else "PASS",
        "blocking": blocking,
        "warnings": len(items) - blocking,
        "checks": checks,
        "findings": items,
        "evidence": evidence or {},
    }


def write_report(workspace: Path, report: dict, *, state: Path | None = None) -> None:
    workspace = Path(workspace)
    marker = workspace / COMPLETION_MARKER_FILENAME
    marker.unlink(missing_ok=True)
    text = yaml.safe_dump(report, sort_keys=False, allow_unicode=True)
    temporary = workspace / f".{REPORT_FILENAME}.{uuid.uuid4().hex}.tmp"
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, workspace / REPORT_FILENAME)
    marker.write_text(json.dumps({"report_sha256": _digest(text.encode()), "status": report["status"]}) + "\n")
    if state is not None:
        (Path(state) / REPORT_FILENAME).write_text(text, encoding="utf-8")


def load_report(workspace: Path) -> dict | None:
    """The finalized report, or None when it is absent, altered or incomplete."""
    workspace = Path(workspace)
    report_path = workspace / REPORT_FILENAME
    marker_path = workspace / COMPLETION_MARKER_FILENAME
    try:
        if report_path.is_symlink() or marker_path.is_symlink():
            return None
        data = report_path.read_bytes()
        marker = json.loads(marker_path.read_text())
        report = yaml.safe_load(data)
    except (OSError, ValueError, yaml.YAMLError):
        return None
    if not isinstance(report, dict) or not isinstance(marker, dict):
        return None
    if marker.get("report_sha256") != _digest(data) or marker.get("status") != report.get("status"):
        return None
    if report.get("qa_report_version") != REPORT_VERSION or report.get("status") not in ("PASS", "FAIL"):
        return None
    blocking = sum(1 for f in report.get("findings", []) if isinstance(f, dict) and f.get("severity") == BLOCKING)
    if (report["status"] == "PASS") != (blocking == 0) or report.get("blocking") != blocking:
        return None
    return report


def qa_report_is_complete(workspace: Any) -> bool:
    return load_report(Path(workspace)) is not None


def report_is_current(report: dict, task_package_sha256: str) -> bool:
    """A report qualifies only the exact task package and the current rubric."""
    return (report.get("task_package_sha256") == task_package_sha256
            and report.get("rubric_version") == RUBRIC_VERSION)
