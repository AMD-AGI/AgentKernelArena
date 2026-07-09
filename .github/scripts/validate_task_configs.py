#!/usr/bin/env python3
"""Validate every benchmark task ``config.yaml`` (CPU-only, pre-merge friendly).

This intentionally has no GPU / heavy dependencies so it can run on a plain
GitHub-hosted runner as part of the pre-merge gate. It mirrors how the framework
discovers tasks (``src/tasks.py`` scans ``tasks/**/config.yaml``) and enforces a
minimal, forward-compatible schema.

For each ``tasks/**/config.yaml`` it checks that:
  * the file is syntactically valid YAML;
  * the top-level document is a mapping;
  * the required key ``task_type`` is present and is one of the known categories;
  * ``prompt``, when present, is a mapping;
  * list-style command fields, when present, are lists.

Usage:
    python .github/scripts/validate_task_configs.py [--tasks-root tasks]

Exit code is 0 when every task config is valid, 1 otherwise. Failures are printed
both as human-readable lines and as GitHub Actions ``::error`` annotations.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

# Known task categories, derived from the current task tree. Extend this set when
# a new task family is introduced so the gate keeps catching typos in task_type.
KNOWN_TASK_TYPES = {
    "flydsl2flydsl",
    "hip2hip",
    "instruction2triton",
    "repository",
    "torch2hip",
    "triton2triton",
}

REQUIRED_KEYS = ("task_type",)

LIST_COMMAND_KEYS = (
    "source_file_path",
    "target_kernel_functions",
    "compile_command",
    "correctness_command",
    "performance_command",
)


def validate_one(path: Path) -> List[str]:
    """Return a list of error messages for a single config (empty == valid)."""
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:  # malformed YAML
        return [f"invalid YAML: {exc}"]

    if not isinstance(data, dict):
        return ["top-level YAML document is not a mapping"]

    errors: List[str] = []

    for key in REQUIRED_KEYS:
        if key not in data:
            errors.append(f"missing required key: {key!r}")

    task_type = data.get("task_type")
    if task_type is not None and task_type not in KNOWN_TASK_TYPES:
        allowed = ", ".join(sorted(KNOWN_TASK_TYPES))
        errors.append(f"unknown task_type {task_type!r} (expected one of: {allowed})")

    prompt = data.get("prompt")
    if prompt is not None and not isinstance(prompt, dict):
        errors.append("'prompt' must be a mapping when present")

    for key in LIST_COMMAND_KEYS:
        value = data.get(key)
        if value is not None and not isinstance(value, list):
            errors.append(f"{key!r} must be a list when present")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks-root",
        default=str(REPO_ROOT / "tasks"),
        help="Root directory that contains task folders (default: <repo>/tasks)",
    )
    args = parser.parse_args()

    tasks_root = Path(args.tasks_root)
    if not tasks_root.is_dir():
        print(f"ERROR: tasks root not found: {tasks_root}", file=sys.stderr)
        return 1

    configs = sorted(tasks_root.glob("**/config.yaml"))
    if not configs:
        print(f"ERROR: no task config.yaml found under {tasks_root}", file=sys.stderr)
        return 1

    failed = 0
    for cfg in configs:
        errors = validate_one(cfg)
        if errors:
            failed += 1
            try:
                rel = cfg.relative_to(REPO_ROOT)
            except ValueError:
                rel = cfg
            for err in errors:
                # GitHub Actions annotation + plain line for local runs.
                print(f"::error file={rel}::{err}")
                print(f"[FAIL] {rel}: {err}", file=sys.stderr)

    ok = len(configs) - failed
    print(f"Validated {len(configs)} task config(s): {ok} ok, {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
