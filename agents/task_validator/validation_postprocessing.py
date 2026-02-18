# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import yaml
import logging
from pathlib import Path
from typing import List, Dict, Any


CHECK_NAMES = [
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
]


def validation_post_processing(workspace_paths: List[str], logger: logging.Logger) -> None:
    """
    Aggregate validation reports from all task workspaces and produce a summary.

    Reads `validation_report.yaml` from each workspace, counts PASS/FAIL/WARN
    per check category, and logs a formatted summary table. Also writes
    `validation_summary.yaml` to the first workspace's parent directory.

    Args:
        workspace_paths: List of workspace directory paths
        logger: Logger instance
    """
    total_tasks = len(workspace_paths)
    reports: List[Dict[str, Any]] = []
    missing_reports: List[str] = []

    # Per-check counters
    check_stats: Dict[str, Dict[str, int]] = {}
    for name in CHECK_NAMES:
        check_stats[name] = {"PASS": 0, "FAIL": 0, "WARN": 0, "TIMEOUT": 0, "SKIP": 0}

    overall_counts = {"PASS": 0, "FAIL": 0, "WARN": 0}

    for workspace_path in workspace_paths:
        workspace = Path(workspace_path)
        report_file = workspace / "validation_report.yaml"

        if not report_file.exists():
            missing_reports.append(str(workspace_path))
            overall_counts["FAIL"] += 1
            continue

        try:
            with open(report_file, 'r') as f:
                report = yaml.safe_load(f)

            if not isinstance(report, dict):
                missing_reports.append(str(workspace_path))
                overall_counts["FAIL"] += 1
                continue

            reports.append(report)

            # Count overall status
            overall_status = report.get("overall_status", "FAIL").upper()
            if overall_status in overall_counts:
                overall_counts[overall_status] += 1
            else:
                overall_counts["FAIL"] += 1

            # Count per-check statuses
            checks = report.get("checks", {})
            for check_name in CHECK_NAMES:
                check = checks.get(check_name, {})
                status = check.get("status", "SKIP").upper() if isinstance(check, dict) else "SKIP"
                if status in check_stats[check_name]:
                    check_stats[check_name][status] += 1
                else:
                    check_stats[check_name]["FAIL"] += 1

        except Exception as e:
            logger.error(f"Error reading validation report from {workspace_path}: {e}")
            missing_reports.append(str(workspace_path))
            overall_counts["FAIL"] += 1

    # Log summary
    logger.info("=" * 90)
    logger.info("Task Validation Summary Report")
    logger.info("=" * 90)
    logger.info(f"Total Tasks:      {total_tasks}")
    logger.info(f"Reports Found:    {len(reports)}")
    logger.info(f"Reports Missing:  {len(missing_reports)}")
    logger.info(f"Overall PASS:     {overall_counts['PASS']}")
    logger.info(f"Overall WARN:     {overall_counts['WARN']}")
    logger.info(f"Overall FAIL:     {overall_counts['FAIL']}")
    logger.info("-" * 90)

    # Per-check breakdown table
    header = f"{'Check':<35} {'PASS':>6} {'FAIL':>6} {'WARN':>6} {'TIMEOUT':>8} {'SKIP':>6}"
    logger.info(header)
    logger.info("-" * 90)
    for check_name in CHECK_NAMES:
        stats = check_stats[check_name]
        row = f"{check_name:<35} {stats['PASS']:>6} {stats['FAIL']:>6} {stats['WARN']:>6} {stats['TIMEOUT']:>8} {stats['SKIP']:>6}"
        logger.info(row)

    logger.info("-" * 90)

    # Per-task detail
    logger.info("Per-Task Results:")
    logger.info("-" * 90)
    for report in reports:
        task_name = report.get("task_name", "unknown")
        overall = report.get("overall_status", "UNKNOWN")
        summary = report.get("summary", "").strip().split("\n")[0]  # First line only
        logger.info(f"  {overall:<6} {task_name:<40} {summary[:60]}")

    if missing_reports:
        logger.info("")
        logger.info("Tasks with missing validation reports:")
        for path in missing_reports:
            logger.info(f"  MISSING  {Path(path).name}")

    logger.info("=" * 90)

    # Write summary YAML
    summary_data = {
        "total_tasks": total_tasks,
        "reports_found": len(reports),
        "reports_missing": len(missing_reports),
        "overall_counts": overall_counts,
        "per_check_stats": {k: dict(v) for k, v in check_stats.items()},
        "task_results": [
            {
                "task_name": r.get("task_name", "unknown"),
                "overall_status": r.get("overall_status", "UNKNOWN"),
                "summary": r.get("summary", ""),
            }
            for r in reports
        ],
        "missing_report_paths": missing_reports,
    }

    # Save to the parent of the first workspace (the workspace_directory)
    if workspace_paths:
        summary_dir = Path(workspace_paths[0]).parent
        summary_file = summary_dir / "validation_summary.yaml"
        try:
            with open(summary_file, 'w') as f:
                yaml.dump(summary_data, f, default_flow_style=False, sort_keys=False)
            logger.info(f"Validation summary written to: {summary_file}")
        except Exception as e:
            logger.error(f"Failed to write validation summary: {e}")
