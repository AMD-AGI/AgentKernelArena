#!/usr/bin/env python3
"""Build a browser-friendly data bundle for the Arena report dashboard."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VISUALIZATION_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = VISUALIZATION_ROOT.parent
DASHBOARD_DIR = VISUALIZATION_ROOT / "frontend" / "dashboard"
OUTPUT_JSON = DASHBOARD_DIR / "data.json"
OUTPUT_JS = DASHBOARD_DIR / "data.js"

STATUS_PATTERN = re.compile(
    r"^(PASS|FAIL|PARTIAL)\s+(\S+)\s+Score:\s*([0-9.]+)\s+Speedup:\s*([0-9.]+)x\s*$"
)


def format_run_timestamp(raw: str) -> str:
    try:
        dt = datetime.strptime(raw, "%Y%m%d_%H%M%S")
    except ValueError:
        return raw
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def load_status_map(report_path: Path) -> dict[str, dict[str, Any]]:
    status_map: dict[str, dict[str, Any]] = {}
    for line in report_path.read_text().splitlines():
        match = STATUS_PATTERN.match(line.strip())
        if not match:
            continue
        status, task_name, score, speedup = match.groups()
        status_map[task_name] = {
            "status": status,
            "score_from_report": float(score),
            "speedup_from_report": float(speedup),
        }
    return status_map


def as_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def artifact_path(path: Path) -> str:
    return "artifacts/" + path.relative_to(PROJECT_ROOT).as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build dashboard data for AgentKernelArena visualization."
    )
    parser.add_argument(
        "--include-workspace-runs",
        action="store_true",
        help="Also scan workspace_*/run_*/reports outside visualization/reports.",
    )
    return parser.parse_args()


def is_local_visualization_report_directory(report_dir: Path) -> bool:
    try:
        relative = report_dir.relative_to(PROJECT_ROOT)
    except ValueError:
        return False

    return (
        len(relative.parts) == 3
        and relative.parts[0] == "visualization"
        and relative.parts[1] == "reports"
    )


def is_workspace_run_report_directory(report_dir: Path) -> bool:
    try:
        relative = report_dir.relative_to(PROJECT_ROOT)
    except ValueError:
        return False

    return (
        len(relative.parts) == 3
        and relative.parts[0].startswith("workspace_")
        and relative.parts[1].startswith("run_")
        and relative.parts[2] == "reports"
    )


def has_required_report_files(report_dir: Path) -> bool:
    return all(
        (report_dir / filename).exists()
        for filename in ("overall_summary.csv", "task_type_breakdown.json", "overall_report.txt")
    )


def discover_report_directories(include_workspace_runs: bool = False) -> list[Path]:
    report_dirs: list[Path] = []
    seen: set[Path] = set()

    visualization_reports_root = VISUALIZATION_ROOT / "reports"
    if visualization_reports_root.exists():
        for report_dir in sorted(p for p in visualization_reports_root.iterdir() if p.is_dir()):
            report_dir = report_dir.resolve()
            if report_dir in seen or not is_local_visualization_report_directory(report_dir):
                continue
            if not has_required_report_files(report_dir):
                continue
            seen.add(report_dir)
            report_dirs.append(report_dir)

    if include_workspace_runs:
        for workspace_dir in sorted(
            p for p in PROJECT_ROOT.iterdir() if p.is_dir() and p.name.startswith("workspace_")
        ):
            for run_dir in sorted(
                p for p in workspace_dir.iterdir() if p.is_dir() and p.name.startswith("run_")
            ):
                report_dir = (run_dir / "reports").resolve()
                if report_dir in seen or not report_dir.is_dir():
                    continue
                if not is_workspace_run_report_directory(report_dir):
                    continue
                if not has_required_report_files(report_dir):
                    continue
                seen.add(report_dir)
                report_dirs.append(report_dir)

    return sorted(report_dirs, key=lambda path: path.relative_to(PROJECT_ROOT).as_posix())


def report_identity(report_dir: Path) -> dict[str, str]:
    relative = report_dir.relative_to(PROJECT_ROOT)
    if is_local_visualization_report_directory(report_dir):
        base_path = relative
    else:
        base_path = relative.parent if relative.name == "reports" else relative
    label = base_path.as_posix() if base_path.parts else relative.as_posix()
    return {
        "id": label.replace("/", "__"),
        "label": label,
        "reportPath": relative.as_posix(),
    }


def build_dataset(include_workspace_runs: bool = False) -> dict[str, Any]:
    warnings: list[str] = []
    reports: list[dict[str, Any]] = []
    task_catalog: dict[str, dict[str, Any]] = {}
    all_statuses: set[str] = set()
    all_gpus: set[str] = set()
    discovered_report_dirs = discover_report_directories(
        include_workspace_runs=include_workspace_runs
    )

    for report_dir in discovered_report_dirs:
        report_meta = report_identity(report_dir)
        summary_csv = report_dir / "overall_summary.csv"
        breakdown_json = report_dir / "task_type_breakdown.json"
        detail_report = report_dir / "overall_report.txt"

        missing = [p.name for p in (summary_csv, breakdown_json, detail_report) if not p.exists()]
        if missing:
            warnings.append(
                f"Skipped {report_dir.name}: missing {', '.join(sorted(missing))}"
            )
            continue

        breakdown = json.loads(breakdown_json.read_text())
        status_map = load_status_map(detail_report)
        tasks: list[dict[str, Any]] = []

        with summary_csv.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                task_name = row["Task Name"].strip()
                task_type = row["Task Type"].strip()
                status = status_map.get(task_name, {}).get("status", "UNKNOWN")
                task = {
                    "taskName": task_name,
                    "taskType": task_type,
                    "status": status,
                    "score": as_float(row["Score"]),
                    "speedup": as_float(row["Speedup"]),
                    "optimizationSummary": row["Optimization_summary"].strip(),
                }
                tasks.append(task)
                all_statuses.add(status)

                catalog_entry = task_catalog.setdefault(
                    task_name,
                    {
                        "taskName": task_name,
                        "taskType": task_type,
                        "presentInReports": [],
                    },
                )
                if catalog_entry["taskType"] != task_type:
                    warnings.append(
                        f"Task type mismatch for {task_name}: "
                        f"{catalog_entry['taskType']} vs {task_type} in {report_meta['label']}"
                    )
                catalog_entry["presentInReports"].append(report_meta["id"])

        missing_in_csv = set(status_map) - {task["taskName"] for task in tasks}
        for task_name in sorted(missing_in_csv):
            status_info = status_map[task_name]
            inferred_task_type = task_name.split("/", 1)[0] if "/" in task_name else "unknown"
            task = {
                "taskName": task_name,
                "taskType": inferred_task_type,
                "status": status_info["status"],
                "score": status_info["score_from_report"],
                "speedup": status_info["speedup_from_report"],
                "optimizationSummary": "Recovered from overall_report.txt",
            }
            tasks.append(task)
            all_statuses.add(task["status"])
            task_catalog.setdefault(
                task_name,
                {
                    "taskName": task_name,
                    "taskType": inferred_task_type,
                    "presentInReports": [],
                },
            )["presentInReports"].append(report_meta["id"])
            warnings.append(
                f"{report_meta['label']}: recovered {task_name} from overall_report.txt only"
            )

        overall = breakdown.get("overall", {})
        task_types = breakdown.get("task_types", {})
        all_gpus.add(str(breakdown.get("target_gpu", "unknown")))

        reports.append(
            {
                "id": report_meta["id"],
                "label": report_meta["label"],
                "agent": breakdown.get("agent", "unknown"),
                "runTimestamp": breakdown.get("run_timestamp", ""),
                "runTimestampFormatted": format_run_timestamp(
                    breakdown.get("run_timestamp", "")
                ),
                "targetGpu": breakdown.get("target_gpu", "unknown"),
                "reportPath": report_meta["reportPath"],
                "overall": overall,
                "taskTypes": task_types,
                "tasks": sorted(tasks, key=lambda item: item["taskName"]),
                "sourceFiles": {
                    "summaryCsv": artifact_path(summary_csv),
                    "breakdownJson": artifact_path(breakdown_json),
                    "overallReport": artifact_path(detail_report),
                },
            }
        )

    reports.sort(
        key=lambda report: report.get("overall", {}).get("total_score", 0.0), reverse=True
    )

    timestamps = [
        report["runTimestamp"]
        for report in reports
        if report.get("runTimestamp")
    ]
    latest_run = max(timestamps) if timestamps else ""
    task_type_totals: dict[str, int] = defaultdict(int)
    for item in task_catalog.values():
        task_type_totals[item["taskType"]] += 1

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    dataset = {
        "meta": {
            "generatedAt": generated_at,
            "latestRunTimestamp": latest_run,
            "latestRunTimestampFormatted": format_run_timestamp(latest_run) if latest_run else "",
            "reportCount": len(reports),
            "taskCount": len(task_catalog),
            "statuses": sorted(all_statuses),
            "targetGpus": sorted(all_gpus),
            "scanRoot": PROJECT_ROOT.as_posix(),
            "includeWorkspaceRuns": include_workspace_runs,
            "discoveredReportDirectories": [
                path.relative_to(PROJECT_ROOT).as_posix() for path in discovered_report_dirs
            ],
            "warnings": warnings,
        },
        "reports": reports,
        "taskCatalog": sorted(task_catalog.values(), key=lambda item: item["taskName"]),
        "taskTypeTotals": dict(sorted(task_type_totals.items())),
    }
    return dataset


def main() -> None:
    args = parse_args()
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset(include_workspace_runs=args.include_workspace_runs)
    OUTPUT_JSON.write_text(json.dumps(dataset, indent=2))
    OUTPUT_JS.write_text(
        "window.ARENA_REPORT_DATA = " + json.dumps(dataset, indent=2) + ";\n"
    )

    print(f"Wrote {OUTPUT_JSON.relative_to(VISUALIZATION_ROOT)}")
    print(f"Wrote {OUTPUT_JS.relative_to(VISUALIZATION_ROOT)}")
    print(
        f"Discovered {len(dataset['reports'])} report directories under {PROJECT_ROOT} "
        f"(include_workspace_runs={args.include_workspace_runs})"
    )
    warnings = dataset["meta"]["warnings"]
    if warnings:
        print(f"Warnings: {len(warnings)}")
        for warning in warnings:
            print(f"  - {warning}")


if __name__ == "__main__":
    main()
