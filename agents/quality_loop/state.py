# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


TERMINAL_TASK_STATES = {
    "completed",
    "reported_failure",
    "platform_deferred",
}

RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def validate_run_id(run_id: str) -> str:
    if not RUN_ID_PATTERN.fullmatch(run_id) or run_id in {".", ".."}:
        raise ValueError(
            "quality_loop run ID must contain only letters, digits, '.', '_', or '-'"
        )
    return run_id


def resolve_worktree(repo_root: Path, stored_path: str | Path) -> Path:
    path = Path(stored_path)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class AuditState:
    """Crash-safe YAML manifest used for resume and publication summaries."""

    def __init__(self, path: Path, data: dict[str, Any]):
        self.path = path
        self.data = data

    @classmethod
    def create(
        cls,
        path: Path,
        *,
        run_id: str,
        config_fingerprint: str,
        repo_slug: str,
        base_sha: str,
        base_branch: str,
        branch: str,
        worktree: Path,
    ) -> "AuditState":
        validate_run_id(run_id)
        state = cls(
            path,
            {
                "schema_version": 1,
                "run_id": run_id,
                "created_at": utc_now(),
                "updated_at": utc_now(),
                "config_fingerprint": config_fingerprint,
                "repo_slug": repo_slug,
                "base_sha": base_sha,
                "base_branch": base_branch,
                "branch": branch,
                "worktree": str(worktree),
                "status": "running",
                "tasks": {},
                "pull_request_url": None,
            },
        )
        state.save()
        return state

    @classmethod
    def load(cls, path: Path) -> "AuditState":
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict) or raw.get("schema_version") != 1:
            raise ValueError(f"unsupported or corrupt quality_loop state: {path}")
        return cls(path, raw)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data["updated_at"] = utc_now()
        tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        tmp.write_text(
            yaml.safe_dump(self.data, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        os.replace(tmp, self.path)

    def task(self, task_id: str) -> dict[str, Any]:
        tasks = self.data.setdefault("tasks", {})
        return tasks.setdefault(
            task_id,
            {
                "state": "pending",
                "events": [],
                "warnings": [],
                "changes": [],
            },
        )

    def transition(self, task_id: str, state: str, **fields: Any) -> None:
        record = self.task(task_id)
        record["state"] = state
        record.update(fields)
        record.setdefault("events", []).append({"at": utc_now(), "state": state})
        self.save()

    def is_terminal(self, task_id: str) -> bool:
        return self.task(task_id).get("state") in TERMINAL_TASK_STATES

    def finish(self, status: str, *, pull_request_url: str | None = None) -> None:
        self.data["status"] = status
        if pull_request_url:
            self.data["pull_request_url"] = pull_request_url
        self.save()
