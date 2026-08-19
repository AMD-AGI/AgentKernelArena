# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Aggregate framework-finalized task-validator reports."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from agents.task_validator.report_schema import (
    CHECK_NAMES,
    REPORT_FILENAME,
    REPORT_SCHEMA_VERSION,
    validation_report_is_complete,
)


def validation_post_processing(
    workspace_paths: list[str], logger: logging.Logger
) -> bool:
    """Write a deterministic summary and return whether every task passed.

    WARN remains a completed, non-failing validation result. Missing, legacy,
    malformed, tampered, TIMEOUT, and ordinary FAIL reports make this gate fail.
    """
    total_tasks = len(workspace_paths)
    reports: list[dict[str, Any]] = []
    invalid_reports: list[str] = []
    check_stats = {
        name: {"PASS": 0, "FAIL": 0, "WARN": 0, "TIMEOUT": 0, "SKIP": 0}
        for name in CHECK_NAMES
    }
    overall_counts = {"PASS": 0, "FAIL": 0, "WARN": 0}

    for workspace_path in workspace_paths:
        workspace = Path(workspace_path)
        report_file = workspace / REPORT_FILENAME
        if not validation_report_is_complete(workspace):
            logger.error(
                "Invalid or incomplete validator report in %s (a version-%s "
                "framework completion marker is required)",
                workspace,
                REPORT_SCHEMA_VERSION,
            )
            invalid_reports.append(str(workspace))
            overall_counts["FAIL"] += 1
            continue

        try:
            with report_file.open() as handle:
                report = yaml.safe_load(handle)
        except Exception as exc:
            logger.error("Error reading validation report from %s: %s", workspace, exc)
            invalid_reports.append(str(workspace))
            overall_counts["FAIL"] += 1
            continue

        reports.append(report)
        overall_status = report["overall_status"]
        overall_counts[overall_status] += 1
        for check_name in CHECK_NAMES:
            check_stats[check_name][report["checks"][check_name]["status"]] += 1

    logger.info("=" * 90)
    logger.info("Task Validation Summary Report (schema v%s)", REPORT_SCHEMA_VERSION)
    logger.info("=" * 90)
    logger.info("Total Tasks:      %s", total_tasks)
    logger.info("Reports Valid:    %s", len(reports))
    logger.info("Reports Invalid:  %s", len(invalid_reports))
    logger.info("Overall PASS:     %s", overall_counts["PASS"])
    logger.info("Overall WARN:     %s", overall_counts["WARN"])
    logger.info("Overall FAIL:     %s", overall_counts["FAIL"])
    logger.info("-" * 90)

    header = f"{'Check':<35} {'PASS':>6} {'FAIL':>6} {'WARN':>6} {'TIMEOUT':>8} {'SKIP':>6}"
    logger.info(header)
    logger.info("-" * 90)
    for check_name in CHECK_NAMES:
        stats = check_stats[check_name]
        logger.info(
            f"{check_name:<35} {stats['PASS']:>6} {stats['FAIL']:>6} "
            f"{stats['WARN']:>6} {stats['TIMEOUT']:>8} {stats['SKIP']:>6}"
        )

    logger.info("-" * 90)
    logger.info("Per-Task Results:")
    logger.info("-" * 90)
    for report in reports:
        summary = report.get("summary", "").strip().split("\n")[0]
        logger.info(
            "  %-6s %-40s %s",
            report["overall_status"],
            report["task_name"],
            summary[:60],
        )

    if invalid_reports:
        logger.info("")
        logger.info("Tasks with invalid/incomplete validation reports:")
        for path in invalid_reports:
            logger.info("  INVALID  %s", Path(path).name)
    logger.info("=" * 90)

    summary_data = {
        "validation_schema_version": REPORT_SCHEMA_VERSION,
        "validation_passed": not invalid_reports and overall_counts["FAIL"] == 0,
        "total_tasks": total_tasks,
        "reports_valid": len(reports),
        "reports_invalid": len(invalid_reports),
        "overall_counts": overall_counts,
        "per_check_stats": check_stats,
        "task_results": [
            {
                "task_name": report["task_name"],
                "overall_status": report["overall_status"],
                "validation_errors": report.get("validation_errors", []),
                "validation_warnings": report.get("validation_warnings", []),
                "policy_findings": report.get("policy_findings", []),
                "summary": report.get("summary", ""),
            }
            for report in reports
        ],
        "invalid_report_paths": invalid_reports,
    }

    if workspace_paths:
        summary_file = Path(workspace_paths[0]).parent / "validation_summary.yaml"
        try:
            with summary_file.open("w") as handle:
                yaml.safe_dump(summary_data, handle, default_flow_style=False, sort_keys=False)
            logger.info("Validation summary written to: %s", summary_file)
        except Exception as exc:
            logger.error("Failed to write validation summary: %s", exc)
            return False

    return bool(workspace_paths) and not invalid_reports and overall_counts["FAIL"] == 0
