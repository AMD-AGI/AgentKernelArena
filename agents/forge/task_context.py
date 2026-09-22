"""Read the framework-owned v2 invocation, never task-local score files."""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import time

from src.task_protocol import CaseManifest, RESULT_PREFIX, parse_command_result
from src.task_spec import TaskSpec


@dataclass(frozen=True)
class TaskContext:
    path: Path
    spec: TaskSpec
    workspace: Path
    baseline_workspace: Path
    manifest: CaseManifest

    @classmethod
    def load(cls, path: str | Path | None = None, *, workspace: Path | None = None):
        value = path or os.environ.get("ARENA_TASK_CONTEXT")
        if not value:
            raise ValueError("Forge requires framework-provided ARENA_TASK_CONTEXT (task schema v2)")
        path = Path(value).resolve(strict=True)
        document = json.loads(path.read_text())
        if not isinstance(document, dict) or type(document.get("version")) is not int or document["version"] != 1:
            raise ValueError("Unsupported ARENA_TASK_CONTEXT version")
        spec = TaskSpec.from_mapping(document["task_config"], task_id=document["task_id"])
        roots = []
        for key in ("workspace", "baseline_workspace"):
            raw = Path(document[key])
            if not raw.is_absolute() or not raw.is_dir():
                raise ValueError(f"ARENA_TASK_CONTEXT {key} must be an existing absolute directory")
            roots.append(raw.resolve(strict=True))
        candidate, baseline = roots
        if candidate == baseline or candidate in baseline.parents or baseline in candidate.parents:
            raise ValueError("Baseline and candidate require independent workspace roots")
        if candidate == path or candidate in path.parents or baseline in path.parents:
            raise ValueError("ARENA_TASK_CONTEXT must be outside task workspaces")
        if workspace is not None and Path(workspace).resolve(strict=True) != candidate:
            raise ValueError("ARENA_TASK_CONTEXT workspace does not match launch workspace")
        result = parse_command_result(RESULT_PREFIX + json.dumps(document["manifest"]),
                                      role="task", action="validate-task", returncode=0)
        return cls(path, spec, candidate, baseline, CaseManifest.from_result(result))


def bounded_spec(spec: TaskSpec, deadline: float) -> TaskSpec:
    """Each bridge action consumes the same total launch budget."""
    remaining = int(deadline - time.time())
    if remaining < 1:
        raise TimeoutError("Forge invocation deadline exhausted")
    return replace(spec, actions=tuple(replace(action, timeout_s=min(action.timeout_s, remaining))
                                       for action in spec.actions))
