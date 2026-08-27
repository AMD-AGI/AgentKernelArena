from __future__ import annotations

import hashlib

import pytest

from src.eval_tools.factory import task_artifact_root


def test_task_artifact_root_uses_workspace_fallback(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("AKA_EVAL_TOOL_ARTIFACT_SCORING_ROOT", raising=False)
    workspace = tmp_path / "task"
    assert task_artifact_root(workspace) == workspace.resolve() / "tool_reports"


def test_task_artifact_root_uses_dedicated_runtime_tree(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "tasks" / "example"
    root = tmp_path / "isolated-artifacts"
    monkeypatch.setenv("AKA_EVAL_TOOL_ARTIFACT_SCORING_ROOT", str(root))
    digest = hashlib.sha256(str(workspace.resolve()).encode()).hexdigest()[:12]
    assert task_artifact_root(workspace) == root / f"example-{digest}"


def test_task_artifact_root_rejects_relative_runtime_root(monkeypatch) -> None:
    monkeypatch.setenv("AKA_EVAL_TOOL_ARTIFACT_SCORING_ROOT", "relative")
    with pytest.raises(ValueError, match="must be absolute"):
        task_artifact_root("/workspace/task")
