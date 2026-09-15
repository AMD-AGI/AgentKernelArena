"""Execute v2 actions with argv, one action deadline, and fresh process evidence."""
from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid

from .runtime_env import PYTHON_ENV_VAR, build_subprocess_env
from .task_protocol import (
    ActionResult, CaseManifest, TaskProtocolError, merge_command_results, parse_command_result,
)
from .task_spec import TaskSpec


class TaskExecutionError(RuntimeError):
    """The action failed to execute or produce valid evidence."""

    def __init__(self, message: str, *, commands: tuple[CommandEvidence, ...] = ()):
        super().__init__(message)
        self.commands = commands


@dataclass(frozen=True)
class CommandEvidence:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    elapsed_s: float


@dataclass(frozen=True)
class ExecutedAction:
    invocation_id: str
    result: ActionResult
    commands: tuple[CommandEvidence, ...]


def _run_process(argv: tuple[str, ...], workspace: Path, env: dict, timeout: float) -> CommandEvidence:
    started = time.monotonic()
    with subprocess.Popen(argv, cwd=workspace, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True, errors="replace",
                          start_new_session=True) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except BaseException as exc:
            # Kill the command's process group, including timed-out compiler/JIT
            # descendants. They must not keep consuming a later action's GPU.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = process.communicate()
            if isinstance(exc, subprocess.TimeoutExpired):
                exc.output = stdout
                exc.stderr = stderr
            raise
    return CommandEvidence(argv, process.returncode, stdout, stderr, time.monotonic() - started)


def run_action(spec: TaskSpec, workspace: Path, *, role: str, action: str,
               phase: str, manifest: CaseManifest | None = None,
               extra_env: dict[str, str] | None = None,
               logger: logging.Logger | None = None) -> ExecutedAction:
    """Execute and validate evidence; lifecycle gating is a separate decision.

    Caller retains manifest in trusted framework memory/storage. This function
    never discovers prior reports in an agent-editable directory. Compile can
    run without a manifest; correctness/performance cannot.
    """
    if phase not in ("task_validation", "candidate_evaluation"):
        raise TaskExecutionError(f"Unknown evaluation phase: {phase}")
    if action in ("correctness", "performance") and manifest is None:
        raise TaskExecutionError("Correctness/performance require an independently captured manifest")
    selected = spec.action(role, action)
    workspace = Path(workspace).resolve(strict=True)
    env = build_subprocess_env()
    # Clear inherited framework context, then set it for this exact invocation.
    for key in list(env):
        if key.startswith("ARENA_"):
            del env[key]
    if extra_env:
        if any(key.startswith("ARENA_") for key in extra_env):
            raise TaskExecutionError("extra_env cannot override framework ARENA_* context")
        env.update({str(key): str(value) for key, value in extra_env.items()})
    env["ARENA_EVAL_PHASE"] = phase
    log = logger or logging.getLogger(__name__)
    invocation_id = uuid.uuid4().hex
    deadline = time.monotonic() + selected.timeout_s
    results = []
    evidence = []
    for command in selected.commands:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TaskExecutionError(f"{role}.{action} exceeded its {selected.timeout_s}s action deadline",
                                     commands=tuple(evidence))
        argv = command
        python = env.get(PYTHON_ENV_VAR)
        if python:
            if command[0] in ("python", "python3"):
                argv = (python,) + command[1:]
            elif command[0] == "pytest":
                argv = (python, "-m", "pytest") + command[1:]
        log.info("Task %s: %s.%s (%s), argv=%r", spec.task_id, role, action, invocation_id, argv)
        try:
            executed = _run_process(argv, workspace, env, remaining)
        except subprocess.TimeoutExpired as exc:
            evidence.append(CommandEvidence(argv, -signal.SIGKILL, exc.stdout or "", exc.stderr or "", remaining))
            raise TaskExecutionError(f"{role}.{action} exceeded its {selected.timeout_s}s action deadline",
                                     commands=tuple(evidence)) from exc
        except OSError as exc:
            raise TaskExecutionError(f"Cannot execute {role}.{action}: {exc}", commands=tuple(evidence)) from exc
        evidence.append(executed)
        try:
            result = parse_command_result(executed.stdout, role=role, action=action,
                                          returncode=executed.returncode)
        except TaskProtocolError as exc:
            raise TaskExecutionError(f"Invalid {role}.{action} evidence: {exc}", commands=tuple(evidence)) from exc
        results.append(result)
        if not result.passed:
            break
    try:
        result = merge_command_results(results)
        if manifest is not None and action != "validate-task":
            manifest.validate(result)
    except TaskProtocolError as exc:
        raise TaskExecutionError(f"Invalid {role}.{action} evidence: {exc}", commands=tuple(evidence)) from exc
    if time.monotonic() > deadline:
        raise TaskExecutionError(f"{role}.{action} exceeded its {selected.timeout_s}s action deadline",
                                 commands=tuple(evidence))
    return ExecutedAction(invocation_id, result, tuple(evidence))
