"""Translate the shared v2 contract without task-family special cases."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

from src.task_protocol import CaseManifest, RESULT_PREFIX, parse_command_result
from src.task_spec import TaskSpec, load_task_spec


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode()


def regular_file(root: Path, relative: str) -> Path:
    from src.task_spec import relative_path

    relative_path(relative)
    path = root
    for part in Path(relative).parts:
        path /= part
        if path.is_symlink():
            raise ValueError("Apex input and delivery paths cannot traverse symlinks")
    if not path.is_file() or path.stat().st_nlink != 1:
        raise ValueError(f"Apex requires a regular file: {relative}")
    return path


def load_context(workspace: Path, task_config: Path) -> tuple[TaskSpec, dict]:
    context_path = os.environ.get("ARENA_TASK_CONTEXT")
    if not context_path:
        raise ValueError("Apex requires the framework's ARENA_TASK_CONTEXT")
    raw = json.loads(Path(context_path).read_text())
    if raw.get("version") != 1 or Path(raw["workspace"]).resolve() != workspace:
        raise ValueError("Apex task context does not match the workspace")
    spec = TaskSpec.from_mapping(raw["task_config"], task_id=raw["task_id"])
    if load_task_spec(task_config, task_id=spec.task_id).to_mapping() != spec.to_mapping():
        raise ValueError("Apex task context does not match the task declaration")
    result = parse_command_result(RESULT_PREFIX + json.dumps(raw["manifest"]),
                                  role="task", action="validate-task", returncode=0)
    CaseManifest.from_result(result)
    candidate = spec.candidate
    if (candidate.language not in {"python", "triton"}
            or candidate.initial_state != "implemented"
            or candidate.initial_language != candidate.language):
        raise ValueError("Apex supports implemented Python/Triton optimization tasks only")
    if any(edit.scope == "tree" for edit in candidate.editable):
        raise ValueError("Apex requires explicit editable files, not tree scopes")
    if not any(entry.symbol for entry in candidate.entrypoints):
        raise ValueError("Apex requires a declared callable entrypoint")
    for edit in candidate.editable:
        regular_file(workspace, edit.path)
    return spec, raw


def build_task(spec: TaskSpec, source: Path, results: Path, job_path: Path,
               options: dict, arch: str) -> dict:
    files = [edit.path for edit in spec.candidate.editable]
    commands = {}
    timeout = 0
    for action in ("compile", "correctness", "performance"):
        timeout += spec.action("candidate", action).timeout_s
        commands[action] = {
            "argv": [sys.executable, str(Path(__file__).with_name("bridge.py")), str(job_path), action],
            "timeout_seconds": timeout,
        }
    # Apex has a bounded objective. Keep full task instructions in the copied
    # task files instead of truncating Arena's expanded hardware prompt.
    instructions = (
        "Optimize the existing kernel in this task workspace. Read config.yaml, README.md if present, "
        "and all files declared by instructions. Preserve every path and callable interface. "
        "Only candidate.editable is mutable; symbol scopes and allow_new_helpers remain binding. "
        "Keep harnesses, reference code, workloads, numerical tolerances and timing unchanged. "
        "Use the supplied compile, correctness and performance commands. "
        "Leave the selected implementation in this workspace. Arena independently evaluates the bundle. "
        "Never write task_result.yaml or validation_report.yaml."
    )
    return {
        "schema_version": 1,
        "task_id": "arena-" + digest(spec.task_id.encode())[:24],
        "workspace": str(source), "results_dir": str(results),
        "instructions": instructions, "language": spec.candidate.language,
        "editable_files": files,
        "target_functions": list(dict.fromkeys(entry.symbol for entry in spec.candidate.entrypoints if entry.symbol)),
        "commands": commands, "gpu_arch": arch, "mode": "optimize_existing",
        "agent_backend": options["backend"],
        "agent_options": {key: options[key] for key in ("model", "effort")},
        "budget": {key: options[key] for key in ("max_iterations", "max_turns", "timeout_seconds")},
        "recipe": {"kind": "python_triton", "recipe_id": "arena-v2",
                   "sha256": digest(canonical(spec.to_mapping())), "provenance": "external_evaluator"},
        "delivery": {"mode": "bundle"},
    }
