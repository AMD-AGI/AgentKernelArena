"""Durable serialization helpers for evaluation-tool results."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import yaml

from .contracts import EvaluationReport


TASK_RESULT_FIELD = "tool_evaluation"
DEFAULT_REPORT_NAME = "summary.json"


def _atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except Exception:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise


def serialize_report(
    report: EvaluationReport,
    *,
    include_output: bool = False,
) -> dict[str, Any]:
    if not isinstance(report, EvaluationReport):
        raise TypeError("report must be an EvaluationReport")
    return report.to_dict(include_output=include_output)


def write_report(
    report: EvaluationReport,
    output: str | Path,
    *,
    include_output: bool = False,
) -> Path:
    """Atomically write a standalone JSON summary.

    ``output`` may name either a JSON file or a directory.  Raw tool output is
    omitted by default; plugins should store large logs in their artifact dirs.
    """

    output_path = Path(output)
    if output_path.suffix.lower() != ".json":
        output_path = output_path / DEFAULT_REPORT_NAME
    payload = serialize_report(report, include_output=include_output)
    _atomic_write_text(
        output_path,
        json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False) + "\n",
    )
    return output_path


def merge_task_result_data(
    task_result: Mapping[str, Any],
    report: EvaluationReport,
) -> dict[str, Any]:
    """Return a copy with the versioned tool result nested under one new key."""

    if not isinstance(task_result, Mapping):
        raise TypeError("task_result must be a mapping")
    merged = dict(task_result)
    merged[TASK_RESULT_FIELD] = serialize_report(report)
    return merged


def merge_task_result_file(
    task_result_path: str | Path,
    report: EvaluationReport,
) -> Path:
    """Atomically merge a report without disturbing legacy scoring fields."""

    path = Path(task_result_path)
    if path.exists():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, Mapping):
            raise ValueError(f"task result is not a mapping: {path}")
    else:
        loaded = {}
    merged = merge_task_result_data(loaded, report)
    _atomic_write_text(
        path,
        yaml.safe_dump(merged, default_flow_style=False, sort_keys=False),
    )
    return path


def has_current_plan(
    task_result: Mapping[str, Any] | str | Path,
    fingerprint: str,
) -> bool:
    """Return whether a result contains a completed report for this exact plan."""

    if isinstance(task_result, (str, Path)):
        path = Path(task_result)
        if not path.is_file():
            return False
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            return False
    else:
        data = task_result
    if not isinstance(data, Mapping):
        return False
    tool_data = data.get(TASK_RESULT_FIELD)
    return bool(
        isinstance(tool_data, Mapping)
        and tool_data.get("plan_fingerprint") == fingerprint
        and isinstance(tool_data.get("tools"), Mapping)
    )


__all__ = [
    "DEFAULT_REPORT_NAME",
    "TASK_RESULT_FIELD",
    "has_current_plan",
    "merge_task_result_data",
    "merge_task_result_file",
    "serialize_report",
    "write_report",
]
