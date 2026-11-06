# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Summarize framework-finalized QA reports for a run."""
from __future__ import annotations

from collections import Counter
import json
import logging
from pathlib import Path

from .report import load_report
from .rubric import BLOCKING


def audit_post_processing(workspace_paths: list[str], logger: logging.Logger) -> bool:
    """Write qa_summary.json beside the workspaces; True only when every task passed."""
    rows, invalid = [], []
    blocking_ids, warning_ids = Counter(), Counter()
    for path in workspace_paths:
        workspace = Path(path)
        report = load_report(workspace)
        if report is None:
            invalid.append(str(workspace))
            continue
        for finding in report["findings"]:
            (blocking_ids if finding["severity"] == BLOCKING else warning_ids)[finding["id"]] += 1
        rows.append({"task_id": report["task_id"], "status": report["status"], "blocking": report["blocking"],
                     "warnings": report["warnings"], "workspace": str(workspace),
                     "blocking_findings": [f["id"] + ": " + f["detail"][:300] for f in report["findings"]
                                           if f["severity"] == BLOCKING]})
    passed = sum(row["status"] == "PASS" for row in rows)
    summary = {"tasks": len(workspace_paths), "pass": passed, "fail": len(rows) - passed,
               "invalid_reports": invalid, "blocking_by_check": dict(blocking_ids),
               "warnings_by_check": dict(warning_ids), "results": sorted(rows, key=lambda r: r["task_id"])}
    if workspace_paths:
        target = Path(workspace_paths[0]).parent / "qa_summary.json"
        target.write_text(json.dumps(summary, indent=2) + "\n")
        logger.info("QA summary written to %s", target)
    logger.info("QA audit: %d/%d PASS, %d FAIL, %d invalid", passed, len(workspace_paths),
                len(rows) - passed, len(invalid))
    for check_id, count in blocking_ids.most_common():
        logger.info("  blocking %-22s %d", check_id, count)
    return not invalid and passed == len(workspace_paths)
