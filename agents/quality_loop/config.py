# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _runtime_root(value: Any, name: str) -> str:
    path = Path(str(value))
    if path.is_absolute() or not path.parts or any(part == ".." for part in path.parts):
        raise ValueError(f"{name} must be a repository-relative path without '..'")
    return path.as_posix()


@dataclass(frozen=True)
class BackendConfig:
    name: str = "codex"
    model: str | None = None
    effort: str = "xhigh"
    timeout_seconds: int = 3600

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "BackendConfig":
        raw = raw or {}
        name = str(raw.get("name", "codex")).strip().lower()
        if name != "codex":
            raise ValueError("quality_loop currently supports only the codex backend")
        timeout = int(raw.get("timeout_seconds", 3600))
        if timeout <= 0:
            raise ValueError("backend.timeout_seconds must be positive")
        model = raw.get("model")
        return cls(
            name=name,
            model=str(model) if model else None,
            effort=str(raw.get("effort", "xhigh")),
            timeout_seconds=timeout,
        )


@dataclass(frozen=True)
class GitHubConfig:
    publish: bool = True
    draft_pr: bool = True
    issue_labels: tuple[str, ...] = ()
    branch_prefix: str = "quality-loop"
    base_branch: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "GitHubConfig":
        raw = raw or {}
        labels = raw.get("issue_labels", [])
        if isinstance(labels, str):
            labels = [labels]
        prefix = str(raw.get("branch_prefix", "quality-loop")).strip(" /-")
        if not prefix:
            raise ValueError("github.branch_prefix must not be empty")
        base = raw.get("base_branch")
        return cls(
            publish=bool(raw.get("publish", True)),
            draft_pr=bool(raw.get("draft_pr", True)),
            issue_labels=tuple(str(label) for label in labels),
            branch_prefix=prefix,
            base_branch=str(base) if base else None,
        )


@dataclass(frozen=True)
class QualityLoopConfig:
    tasks: tuple[str, ...] = ("all",)
    target_gpu_model: str = "MI300"
    backend: BackendConfig = field(default_factory=BackendConfig)
    reviewer: BackendConfig = field(default_factory=BackendConfig)
    github: GitHubConfig = field(default_factory=GitHubConfig)
    max_repair_attempts: int = 1
    optimization_iterations: int = 1
    easy_speedup_threshold: float = 5.0
    easy_confirmation_runs: int = 3
    case_enhancement: bool = True
    artifact_root: str = "quality_loop_runs"
    worktree_root: str = ".quality_loop_worktrees"
    promotion_task_types: tuple[str, ...] = (
        "hip2hip",
        "triton2triton",
        "flydsl2flydsl",
    )

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "QualityLoopConfig":
        tasks = raw.get("tasks", ["all"])
        if isinstance(tasks, str):
            tasks = [tasks]
        if not isinstance(tasks, list) or not tasks:
            raise ValueError("tasks must be a non-empty string or list")
        target = str(raw.get("target_gpu_model", "")).strip()
        if not target:
            raise ValueError("target_gpu_model is required")

        audit = raw.get("quality_loop", {}) or {}
        if not isinstance(audit, dict):
            raise ValueError("quality_loop must be a mapping")
        iterations = int(audit.get("optimization_iterations", 1))
        if iterations != 1:
            raise ValueError(
                "quality_loop enforces exactly one optimization iteration; "
                "optimization_iterations must be 1"
            )
        repair_attempts = int(audit.get("max_repair_attempts", 1))
        if repair_attempts < 0 or repair_attempts > 1:
            raise ValueError("max_repair_attempts must be 0 or 1")
        confirmations = int(audit.get("easy_confirmation_runs", 3))
        if confirmations < 1:
            raise ValueError("easy_confirmation_runs must be at least 1")
        threshold = float(audit.get("easy_speedup_threshold", 5.0))
        if threshold <= 1.0:
            raise ValueError("easy_speedup_threshold must be greater than 1.0")

        promotion_types = audit.get(
            "promotion_task_types",
            ["hip2hip", "triton2triton", "flydsl2flydsl"],
        )
        return cls(
            tasks=tuple(str(task) for task in tasks),
            target_gpu_model=target,
            backend=BackendConfig.from_dict(audit.get("backend")),
            reviewer=BackendConfig.from_dict(audit.get("reviewer", audit.get("backend"))),
            github=GitHubConfig.from_dict(audit.get("github")),
            max_repair_attempts=repair_attempts,
            optimization_iterations=iterations,
            easy_speedup_threshold=threshold,
            easy_confirmation_runs=confirmations,
            case_enhancement=bool(audit.get("case_enhancement", True)),
            artifact_root=_runtime_root(
                audit.get("artifact_root", "quality_loop_runs"),
                "quality_loop.artifact_root",
            ),
            worktree_root=_runtime_root(
                audit.get("worktree_root", ".quality_loop_worktrees"),
                "quality_loop.worktree_root",
            ),
            promotion_task_types=tuple(str(value) for value in promotion_types),
        )


def load_config(path: Path) -> QualityLoopConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"quality_loop config must be a mapping: {path}")
    return QualityLoopConfig.from_dict(raw)
