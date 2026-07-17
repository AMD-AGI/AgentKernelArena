#!/usr/bin/env python3
"""Materialize canonical perf helpers into task workspaces.

This is a local debugging utility. Normal experiment runs call
setup_workspace(), which materializes helpers automatically.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import shutil
import sys


ROOT = pathlib.Path(__file__).resolve().parents[2]
TASKS_ROOT = ROOT / "tasks"
sys.path.insert(0, str(ROOT))

from src.perf_helper_materialization import materialize_perf_helpers_in_workspace  # noqa: E402


def _copy_task(task: pathlib.Path, out_root: pathlib.Path, force: bool) -> pathlib.Path:
    task = task.resolve()
    out_root = out_root.resolve()

    if not task.exists() or not task.is_dir():
        raise FileNotFoundError(f"task directory not found: {task}")

    try:
        rel = task.relative_to(TASKS_ROOT)
        dest = out_root / rel
    except ValueError:
        dest = out_root / task.name

    if dest.exists():
        if not force:
            raise FileExistsError(f"destination already exists: {dest} (use --force or choose --out)")
        shutil.rmtree(dest)

    shutil.copytree(
        task,
        dest,
        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"),
    )
    return dest


def _materialize(paths: list[pathlib.Path]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for workspace in paths:
        workspace = workspace.resolve()
        if not workspace.exists() or not workspace.is_dir():
            raise FileNotFoundError(f"workspace directory not found: {workspace}")
        materialized = materialize_perf_helpers_in_workspace(workspace)
        if materialized:
            print(f"materialized {workspace}")
        else:
            print(f"no perf helper targets found or already materialized: {workspace}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    workspace_parser = subparsers.add_parser(
        "workspace",
        help="materialize canonical perf helpers into existing workspace directories",
    )
    workspace_parser.add_argument("workspaces", nargs="+", type=pathlib.Path)

    task_parser = subparsers.add_parser(
        "task",
        help="copy task source directories to --out and materialize helpers there",
    )
    task_parser.add_argument("tasks", nargs="+", type=pathlib.Path)
    task_parser.add_argument("--out", required=True, type=pathlib.Path)
    task_parser.add_argument("--force", action="store_true", help="replace existing copied task directories")

    args = parser.parse_args()

    if args.command == "workspace":
        return _materialize(args.workspaces)

    copied = [_copy_task(task, args.out, args.force) for task in args.tasks]
    return _materialize(copied)


if __name__ == "__main__":
    raise SystemExit(main())
