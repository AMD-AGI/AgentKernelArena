"""Read the framework-owned invocation for a schema-v3 task, never task-local score files."""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import time

from src.task_contract import SOLUTION_DIR
from src.task_protocol import CaseManifest, RESULT_PREFIX, parse_command_result
from src.task_spec import TaskSpec


def require_kernel_task(spec: TaskSpec) -> dict:
    """Forge optimizes schema-v3 kernel tasks: one ``solution/`` tree and its ``run`` entry.

    The resolved contract record names the run's target language and the
    baseline's language. Schema-v2 packages (the e2e serving tasks) declare a
    backend-neutral candidate that no Forge kernel backend implements.
    """
    contract = spec.contract
    if contract is None:
        raise ValueError(f"Forge runs schema-v3 kernel tasks; {spec.task_id} has no resolved task.yaml contract")
    candidate = spec.candidate
    entry = "kernel.hip" if candidate.language == "hip" else "kernel.py"
    if (len(candidate.editable) != 1 or candidate.editable[0].path != SOLUTION_DIR
            or candidate.editable[0].scope != "tree"
            or [(e.file, e.kind, e.symbol) for e in candidate.entrypoints]
            != [(f"{SOLUTION_DIR}/{entry}", "function", "run")]):
        raise ValueError(f"Forge expects the schema-v3 candidate {SOLUTION_DIR}/{entry}::run as its only "
                         "editable tree")
    return contract


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
            raise ValueError("Forge requires the framework-provided ARENA_TASK_CONTEXT")
        path = Path(value).resolve(strict=True)
        document = json.loads(path.read_text())
        if not isinstance(document, dict) or type(document.get("version")) is not int or document["version"] != 1:
            raise ValueError("Unsupported ARENA_TASK_CONTEXT version")
        spec = TaskSpec.from_mapping(document["task_config"], task_id=document["task_id"])
        require_kernel_task(spec)
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

    @property
    def target_language(self) -> str:
        return self.spec.contract["target_language"]

    @property
    def baseline_language(self) -> str:
        return self.spec.contract["baseline_language"]

    @property
    def operator(self) -> str:
        return self.spec.contract["operator"]["name"]


def bounded_spec(spec: TaskSpec, deadline: float) -> TaskSpec:
    """Each bridge action consumes the same total launch budget."""
    remaining = int(deadline - time.time())
    if remaining < 1:
        raise TimeoutError("Forge invocation deadline exhausted")
    return replace(spec, actions=tuple(replace(action, timeout_s=min(action.timeout_s, remaining))
                                       for action in spec.actions))
