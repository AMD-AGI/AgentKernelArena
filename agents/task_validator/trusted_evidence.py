"""Capture framework task-session evidence before starting an untrusted backend.

A context file is a transport, not an authority the finalizer discovers. Callers
must pass a snapshot taken from TaskSession memory or load the external context
before launching the model. Nothing here reads agent-produced gate decisions.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from src.task_protocol import (
    CaseManifest, baseline_correctness_accepted, merge_command_results, parse_command_result,
)
from src.task_spec import ACTIONS, TaskSpec


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class TrustedTaskEvidence:
    """An immutable value copy; the backend never receives this Python object."""
    serialized: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.serialized.encode()).hexdigest()

    def to_mapping(self) -> dict:
        return json.loads(self.serialized)


def snapshot_task_evidence(value: Mapping | TrustedTaskEvidence, *, task_id: str,
                           workspace: str | Path | None = None,
                           task_config: Mapping | None = None) -> TrustedTaskEvidence:
    if isinstance(value, TrustedTaskEvidence):
        data = value.to_mapping()
    elif isinstance(value, Mapping):
        data = json.loads(_json(dict(value)))
    else:
        raise ValueError("Trusted task evidence must be a framework snapshot, not a path")
    if type(data.get("version")) is not int or data["version"] != 1:
        raise ValueError("Task validation context requires version: 1")
    if data.get("task_id") != task_id:
        raise ValueError("Task validation context task_id does not match this task")
    spec = TaskSpec.from_mapping(data.get("task_config"), task_id=task_id)
    if task_config is not None and spec.to_mapping() != TaskSpec.from_mapping(
        task_config, task_id=task_id
    ).to_mapping():
        raise ValueError("Task validation context config differs from the launched task")
    for key in ("workspace", "baseline_workspace"):
        if not isinstance(data.get(key), str) or not Path(data[key]).is_absolute():
            raise ValueError(f"Context {key} must identify the framework's absolute workspace")
    candidate_root = Path(data["workspace"]).resolve()
    baseline_root = Path(data["baseline_workspace"]).resolve()
    if candidate_root.is_relative_to(baseline_root) or baseline_root.is_relative_to(candidate_root):
        raise ValueError("Baseline and candidate workspaces must be separate")
    if workspace is not None and candidate_root != Path(workspace).resolve():
        raise ValueError("Task validation context workspace does not match this invocation")
    guard = data.get("harness")
    if (not isinstance(guard, dict) or type(guard.get("enforced_during_optimization")) is not bool
            or not isinstance(guard.get("protected_paths"), list)
            or any(not isinstance(path, str) or not path for path in guard.get("protected_paths", []))):
        raise ValueError("Task validation context requires complete framework harness facts")
    if not isinstance(data.get("initial_validation"), dict) or not isinstance(data.get("actions"), list):
        raise ValueError("Task validation context requires initial_validation and actions")
    return TrustedTaskEvidence(_json(data))


def load_task_evidence(path: str | Path, *, task_id: str, workspace: str | Path,
                       task_config: Mapping) -> TrustedTaskEvidence:
    """Load only before backend launch. The context must be outside both workspaces."""
    path = Path(path).absolute()
    resolved = path.resolve(strict=True)
    if path != resolved or not resolved.is_file():
        raise ValueError("Validation context must be a regular file without symlink components")
    if resolved.is_relative_to(Path(workspace).resolve()):
        raise ValueError("Validation context cannot be loaded from the agent workspace")
    snapshot = snapshot_task_evidence(json.loads(resolved.read_text()), task_id=task_id,
                                      workspace=workspace, task_config=task_config)
    if resolved.is_relative_to(Path(snapshot.to_mapping()["baseline_workspace"]).resolve()):
        raise ValueError("Validation context cannot be loaded from the baseline workspace")
    return snapshot


def _argv_matches(actual: list, declared: tuple[str, ...]) -> bool:
    # task_execution substitutes only the framework interpreter. Keep every
    # remaining argument exact, including literal whitespace and shell syntax.
    if actual == list(declared):
        return True
    if not actual or not isinstance(actual[0], str) or not Path(actual[0]).is_absolute():
        return False
    if declared[0] in ("python", "python3"):
        return actual[1:] == list(declared[1:])
    if declared[0] == "pytest":
        return actual[1:] == ["-m", "pytest", *declared[1:]]
    return False


def evaluate_task_evidence(snapshot: TrustedTaskEvidence) -> dict:
    """Recheck command/manifest/policy evidence; preserve every original FAIL.

    TaskSession's accepted flag is necessary but never sufficient. A malformed,
    truncated, contradictory, or wrong-phase context cannot grant an exception.
    """
    data = snapshot.to_mapping()
    spec = TaskSpec.from_mapping(data["task_config"], task_id=data["task_id"])
    errors = []
    records = {}
    results = {}
    ids = set()
    for index, record in enumerate(data["actions"]):
        try:
            if not isinstance(record, dict) or record.get("phase") != "task_validation":
                raise ValueError("Only initial task_validation actions are permitted")
            result_data = record.get("result", {})
            role = record.get("role", result_data.get("role"))
            action = record.get("action", result_data.get("action"))
            key = (role, action)
            if key not in ACTIONS or key in records:
                raise ValueError("Unknown or duplicate action evidence")
            records[key] = record
            if record.get("execution_error"):
                raise ValueError(str(record["execution_error"]))
            invocation = record.get("invocation_id")
            if not isinstance(invocation, str) or not invocation or invocation in ids:
                raise ValueError("Action requires a unique framework invocation_id")
            ids.add(invocation)
            commands = record.get("commands")
            declared = spec.action(role, action)
            if not isinstance(commands, list) or not commands or len(commands) > len(declared.commands):
                raise ValueError("Missing or extra command execution evidence")
            parsed = []
            elapsed = 0.0
            for command_index, command in enumerate(commands):
                if not isinstance(command, dict) or not _argv_matches(command.get("argv", []), declared.commands[command_index]):
                    raise ValueError("Executed argv differs from declared action")
                duration = command.get("elapsed_s")
                if type(duration) not in (int, float) or not math.isfinite(duration) or duration < 0:
                    raise ValueError("Invalid command elapsed_s")
                elapsed += duration
                if type(command.get("returncode")) is not int or not all(
                    isinstance(command.get(field), str) for field in ("stdout", "stderr")
                ):
                    raise ValueError("Command returncode/stdout/stderr are required")
                if parsed and not parsed[-1].passed:
                    raise ValueError("Action continued after a failed command")
                parsed.append(parse_command_result(command["stdout"], role=role, action=action,
                                                   returncode=command["returncode"]))
            result = merge_command_results(parsed)
            if result.passed and len(commands) != len(declared.commands):
                raise ValueError("Passing action omitted configured commands")
            if elapsed > declared.timeout_s:
                raise ValueError("Action exceeded configured timeout")
            if result.to_mapping() != result_data:
                raise ValueError("Recorded result contradicts actual command stdout")
            results[key] = result
        except (ValueError, TypeError, AttributeError, KeyError) as exc:
            errors.append(f"actions[{index}]: {exc}")

    initial = data["initial_validation"]
    for key in ("accepted", "baseline_diagnostic"):
        if type(initial.get(key)) is not bool:
            errors.append(f"initial_validation.{key} must be boolean")
    if not isinstance(initial.get("errors"), list) or any(not isinstance(e, str) for e in initial.get("errors", [])):
        errors.append("initial_validation.errors must be a list of strings")
    else:
        errors.extend(initial["errors"])
    if initial.get("accepted") is not True:
        errors.append("TaskSession rejected initial validation")
    if initial.get("candidate_initial_state") != spec.candidate.initial_state:
        errors.append("Initial candidate state disagrees with task declaration")

    manifest = None
    task = results.get(("task", "validate-task"))
    if task:
        try:
            manifest = CaseManifest.from_result(task)
            states = {m.get("candidate_state") for m in (task.metadata or {}).get("commands", [])
                      if isinstance(m, dict) and "candidate_state" in m}
            if states != {spec.candidate.initial_state}:
                errors.append("validate-task did not confirm actual candidate initial state")
        except ValueError as exc:
            errors.append(str(exc))
    else:
        errors.append("Missing validate-task result")
    diagnostic = False
    baseline_correctness = results.get(("baseline", "correctness"))
    for key, result in results.items():
        if key[0] == "task":
            continue
        try:
            if manifest is None:
                raise ValueError("Action has no independent manifest")
            manifest.validate(result)
            accepted = result.passed
            if key == ("baseline", "correctness"):
                accepted = baseline_correctness_accepted(result, baseline=spec.baseline,
                                                        phase="task_validation", manifest=manifest)
                diagnostic = accepted and not result.passed
            if not accepted:
                errors.append(f"{key[0]}.{key[1]} failed: {result.reason}")
        except ValueError as exc:
            errors.append(f"{key[0]}.{key[1]}: {exc}")
    for action in ("compile", "correctness", "performance"):
        if ("baseline", action) not in results:
            errors.append(f"Missing baseline.{action} evidence")
    numerical = baseline_correctness.status if baseline_correctness else "NOT_RUN"
    if initial.get("baseline_numerical_status") != numerical:
        errors.append("Initial baseline numerical status contradicts execution evidence")
    if initial.get("baseline_diagnostic") != diagnostic:
        errors.append("Initial diagnostic decision contradicts baseline policy/evidence")

    unimplemented = spec.candidate.initial_state == "unimplemented"
    expected_checks = ("candidate_unimplemented" if unimplemented else
                       "verified_as_frozen_baseline" if spec.baseline.kind == "initial_candidate" else "PASS")
    if initial.get("candidate_checks") != expected_checks:
        errors.append("Initial candidate checks are incomplete or inconsistent")
    if unimplemented or spec.baseline.kind == "initial_candidate":
        if any(key[0] == "candidate" for key in records):
            errors.append("Unexpected candidate execution for this initial lifecycle")
    else:
        for action in ("compile", "correctness", "performance"):
            if ("candidate", action) not in results:
                errors.append(f"Missing initial candidate.{action} evidence")
    expected_order = [("task", "validate-task"), *(("baseline", a) for a in ("compile", "correctness", "performance"))]
    if not unimplemented and spec.baseline.kind != "initial_candidate":
        expected_order.extend(("candidate", a) for a in ("compile", "correctness", "performance"))
    if list(records) != expected_order:
        errors.append("Initial actions are missing or out of lifecycle order")
    return {"accepted": not errors, "errors": errors, "diagnostic": diagnostic,
            "baseline_numerical_status": numerical,
            "candidate_unimplemented": unimplemented and not errors,
            "candidate_checks": expected_checks, "results": results, "records": records,
            "spec": spec, "manifest": manifest}
