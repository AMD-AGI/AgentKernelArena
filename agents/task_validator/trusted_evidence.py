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
from src.task_spec import TaskSpec


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
    """Separate a faithfully recorded failed task from broken execution evidence.

    A valid lifecycle may stop at its first failed action. That remains a task
    FAIL, but it is reviewable/repairable; omitted successful actions, altered
    stdout, wrong argv and contradictory lifecycle claims are framework errors.
    """
    data = snapshot.to_mapping()
    spec = TaskSpec.from_mapping(data["task_config"], task_id=data["task_id"])
    errors, failures = [], []
    records, results, ids = {}, {}, set()
    manifest = None
    state_verified = False
    diagnostic = False
    order = [("task", "validate-task"), *(("baseline", a) for a in ("compile", "correctness", "performance"))]
    if spec.candidate.initial_state == "implemented" and spec.baseline.kind == "provided":
        order.extend(("candidate", a) for a in ("compile", "correctness", "performance"))
    stopped = False
    for index, record in enumerate(data["actions"]):
        try:
            if not isinstance(record, dict) or record.get("phase") != "task_validation":
                raise ValueError("Only initial task_validation actions are permitted")
            if stopped:
                raise ValueError("Initial actions continued after a failed action")
            result_data = record.get("result", {})
            role = record.get("role", result_data.get("role"))
            action = record.get("action", result_data.get("action"))
            key = (role, action)
            if index >= len(order) or key != order[index] or key in records:
                raise ValueError("Initial actions are missing, duplicated or out of lifecycle order")
            records[key] = record
            execution_error = record.get("execution_error")
            if execution_error is not None and (not isinstance(execution_error, str) or not execution_error):
                raise ValueError("Invalid execution_error evidence")
            invocation = record.get("invocation_id")
            if not execution_error:
                if not isinstance(invocation, str) or not invocation or invocation in ids:
                    raise ValueError("Action requires a unique framework invocation_id")
                ids.add(invocation)
            commands = record.get("commands")
            declared = spec.action(role, action)
            if not isinstance(commands, list) or len(commands) > len(declared.commands) or (not commands and not execution_error):
                raise ValueError("Missing or extra command execution evidence")
            parsed, elapsed = [], 0.0
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
                try:
                    parsed.append(parse_command_result(command["stdout"], role=role, action=action,
                                                       returncode=command["returncode"]))
                except ValueError:
                    # A runner can crash or violate its result protocol. Its
                    # final failed command is still preserved execution evidence.
                    if not execution_error or command_index != len(commands) - 1:
                        raise
            if execution_error:
                if "result" in record:
                    raise ValueError("Execution error cannot also claim a completed result")
                failures.append(f"{role}.{action}: {execution_error}")
                stopped = True
                continue
            result = merge_command_results(parsed)
            if result.passed and len(commands) != len(declared.commands):
                raise ValueError("Passing action omitted configured commands")
            if elapsed > declared.timeout_s:
                raise ValueError("Action exceeded configured timeout without an execution error")
            if result.to_mapping() != result_data:
                raise ValueError("Recorded result contradicts actual command stdout")
            results[key] = result
            accepted = result.passed
            if role == "task":
                if result.passed:
                    manifest = CaseManifest.from_result(result)
                    states = {m.get("candidate_state") for m in (result.metadata or {}).get("commands", [])
                              if isinstance(m, dict) and "candidate_state" in m}
                    state_verified = states == {spec.candidate.initial_state}
                    if not state_verified:
                        accepted = False
                        failures.append("validate-task did not confirm actual candidate initial state")
            else:
                if manifest is None:
                    raise ValueError("Action has no independent manifest")
                manifest.validate(result)
                if key == ("baseline", "correctness"):
                    accepted = baseline_correctness_accepted(result, baseline=spec.baseline,
                                                            phase="task_validation", manifest=manifest)
                    diagnostic = accepted and not result.passed
            if not accepted:
                failures.append(f"{role}.{action} failed: {result.reason or 'initial-state check failed'}")
                stopped = True
        except (ValueError, TypeError, AttributeError, KeyError) as exc:
            errors.append(f"actions[{index}]: {exc}")
            stopped = True

    complete = list(records) == order
    if not complete and not failures:
        errors.append("Initial actions are missing without a recorded task failure")
    initial = data["initial_validation"]
    for key in ("accepted", "baseline_diagnostic"):
        if type(initial.get(key)) is not bool:
            errors.append(f"initial_validation.{key} must be boolean")
    initial_errors = initial.get("errors")
    if not isinstance(initial_errors, list) or any(not isinstance(e, str) or not e for e in initial_errors):
        errors.append("initial_validation.errors must be a list of nonempty strings")
    elif bool(initial_errors) != bool(failures or errors):
        errors.append("Initial error summary contradicts execution evidence")
    if initial.get("candidate_initial_state") != spec.candidate.initial_state:
        errors.append("Initial candidate state disagrees with task declaration")
    baseline_correctness = results.get(("baseline", "correctness"))
    numerical = baseline_correctness.status if baseline_correctness else "NOT_RUN"
    if initial.get("baseline_numerical_status") != numerical:
        errors.append("Initial baseline numerical status contradicts execution evidence")
    if initial.get("baseline_diagnostic") != diagnostic:
        errors.append("Initial diagnostic decision contradicts baseline policy/evidence")
    unimplemented = spec.candidate.initial_state == "unimplemented"
    expected_checks = ("candidate_unimplemented" if unimplemented else
                       "verified_as_frozen_baseline" if spec.baseline.kind == "initial_candidate" else "PASS")
    if failures or errors or not complete:
        expected_checks = "NOT_RUN"
    if initial.get("candidate_checks") != expected_checks:
        errors.append("Initial candidate checks are incomplete or inconsistent")
    accepted = complete and not errors and not failures
    if initial.get("accepted") != accepted:
        errors.append("TaskSession acceptance contradicts execution evidence")
    return {"accepted": accepted and not errors, "errors": errors, "task_failures": failures,
            "evidence_valid": not errors, "diagnostic": diagnostic,
            "baseline_numerical_status": numerical,
            "candidate_unimplemented": unimplemented and state_verified and not errors,
            "candidate_checks": expected_checks, "results": results, "records": records,
            "spec": spec, "manifest": manifest}
