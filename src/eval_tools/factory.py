"""Default wiring for built-in evaluation tools.

Keeping construction here avoids import-time plugin registration and makes unit
tests free to inject a local/fake runtime.  Production evaluation uses one Unix
socket sidecar per enabled tool.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

from .manager import EvalToolManager
from .plugins import register_builtin_plugins
from .registry import ToolRegistry
from .runtime_client import SidecarRuntimeClient


def create_default_manager() -> EvalToolManager:
    registry = ToolRegistry()
    register_builtin_plugins(registry)
    return EvalToolManager(registry, SidecarRuntimeClient())


def task_artifact_root(workspace: str | Path) -> Path:
    """Return a collision-resistant report root for one task.

    The host runner exposes a dedicated evaluation-tool tree when sidecars are
    active.  Keeping reports there prevents the sidecars' writable ``/artifacts``
    mount from aliasing the complete ``experiments`` tree or task workspace.
    Local/unit-test use keeps the historical workspace-local fallback.
    """

    workspace_path = Path(workspace).resolve(strict=False)
    configured = os.environ.get("AKA_EVAL_TOOL_ARTIFACT_SCORING_ROOT")
    if not configured:
        return workspace_path / "tool_reports"
    root = Path(configured)
    if not root.is_absolute():
        raise ValueError("AKA_EVAL_TOOL_ARTIFACT_SCORING_ROOT must be absolute")
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", workspace_path.name).strip("._-")
    digest = hashlib.sha256(str(workspace_path).encode("utf-8")).hexdigest()[:12]
    return root.resolve(strict=False) / f"{label or 'task'}-{digest}"


__all__ = ["create_default_manager", "task_artifact_root"]
