# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Task validation backend orchestration; evidence is captured before launch."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import uuid
from typing import Any

import yaml

from agents import register_agent
from agents.prompt_input import prompt_input
from .report_schema import finalize_report
from .trusted_evidence import load_task_evidence
from .validation_prompt import build_validation_prompt
from src.runtime_env import PYTHON_ENV_VAR


@dataclass(frozen=True)
class BackendResult:
    output: str
    returncode: int | None
    timed_out: bool
    error: str | None = None


def _stop_process(process: subprocess.Popen) -> None:
    """Terminate this invocation's entire process group, including tool children."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        if sig == signal.SIGTERM:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
    process.wait(timeout=10)


def _run_backend(cmd: list[str], *, backend: str, workspace: str, timeout_seconds: int,
                 logger: logging.Logger, env: dict | None = None,
                 stdin=subprocess.DEVNULL) -> BackendResult:
    if not shutil.which(cmd[0]):
        raise RuntimeError(f"Command {cmd[0]!r} not found; install/authenticate the validator backend")
    # Only argv metadata is logged here; never interpolate shell commands.
    logger.info("Launching validator backend %s in %s", backend, workspace)
    process = subprocess.Popen(cmd, stdin=stdin, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, errors="replace", cwd=workspace,
                               env=env, bufsize=1, start_new_session=True)
    stdout, stderr = [], []
    failed = threading.Event()
    completed = threading.Event()

    def read_stream(stream, output, is_stdout):
        try:
            for line in stream:
                output.append(line.rstrip("\n"))
                if is_stdout:
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if not isinstance(event, dict):
                        continue
                    kind = event.get("type")
                    if backend == "codex":
                        if kind in ("turn.failed", "thread.failed", "error"):
                            failed.set()
                        elif kind == "turn.completed":
                            completed.set()
                    elif kind == "result":
                        if event.get("is_error") is True or event.get("subtype") != "success":
                            failed.set()
                        else:
                            completed.set()
                    if kind in ("turn.completed", "turn.failed", "result"):
                        logger.info("Validator %s event: %s", backend, kind)
        finally:
            stream.close()

    readers = [threading.Thread(target=read_stream, args=(process.stdout, stdout, True), daemon=True),
               threading.Thread(target=read_stream, args=(process.stderr, stderr, False), daemon=True)]
    for reader in readers:
        reader.start()
    timed_out = False
    try:
        process.wait(timeout=timeout_seconds if timeout_seconds > 0 else None)
    except subprocess.TimeoutExpired:
        timed_out = True
        _stop_process(process)
    except BaseException:
        _stop_process(process)
        raise
    for reader in readers:
        reader.join(timeout=5)
    error = None
    if any(reader.is_alive() for reader in readers):
        # A child holding a pipe open is still this invocation's process. It must
        # not leak work into the next task even if the CLI parent already exited.
        _stop_process(process)
        for reader in readers:
            reader.join(timeout=5)
        error = "Validator stream did not close after backend exit"
    if failed.is_set():
        error = "Validator backend emitted a terminal failure event"
    elif not completed.is_set() and not timed_out:
        error = error or "Validator backend did not emit a successful terminal event"
    output = "\n".join(stdout)
    if stderr:
        output += "\n=== STDERR ===\n" + "\n".join(stderr)
    return BackendResult(output, process.returncode, timed_out, error)


def _launch_codex(prompt: str, workspace: str, timeout_seconds: int, logger: logging.Logger,
                  model: str | None = None, effort: str | None = None) -> BackendResult:
    cmd = ["codex", "exec", "--json", "--dangerously-bypass-approvals-and-sandbox",
           "--skip-git-repo-check", "--ephemeral", "-c", "features.memories=false", "--cd", workspace]
    if model:
        cmd.extend(["--model", model])
    if effort:
        cmd.extend(["-c", f"model_reasoning_effort={json.dumps(effort)}"])
    cmd.extend(["--", "-"])
    with prompt_input(prompt) as stream:
        return _run_backend(cmd, backend="codex", workspace=workspace,
                            timeout_seconds=timeout_seconds, logger=logger, stdin=stream)


def _launch_claude_code(prompt: str, workspace: str, timeout_seconds: int, logger: logging.Logger,
                        model: str | None = None, effort: str | None = None,
                        max_budget_usd: float | None = None) -> BackendResult:
    cmd = ["claude", "--print", "--verbose", "--output-format", "stream-json",
           "--include-partial-messages", "--permission-mode", "bypassPermissions", "--no-session-persistence"]
    if model:
        cmd.extend(["--model", model])
    if effort:
        cmd.extend(["--effort", effort])
    if max_budget_usd is not None:
        if type(max_budget_usd) not in (int, float) or not math.isfinite(max_budget_usd) or max_budget_usd <= 0:
            raise ValueError("max_budget_usd must be a positive finite number")
        cmd.extend(["--max-budget-usd", str(max_budget_usd)])
    cmd.extend(["--input-format", "text"])
    env = dict(os.environ, IS_SANDBOX="1", CLAUDE_CODE_DISABLE_AUTO_MEMORY="1")
    with prompt_input(prompt) as stream:
        return _run_backend(cmd, backend="claude_code", workspace=workspace, timeout_seconds=timeout_seconds,
                            logger=logger, env=env, stdin=stream)


def _positive_timeout(value: Any, fallback: int) -> int:
    if isinstance(value, bool):
        return fallback
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed > 0 else fallback


def _resolve_validation_timeouts(
    task_config: dict[str, Any], agent_config: dict[str, Any]
) -> tuple[int, int, int, int]:
    """Return compile, correctness, performance, and backend timeouts.

    Task-level command limits are the evaluator contract and therefore override
    validator defaults. The backend must have enough time to run the commands
    sequentially plus perform its static review.
    """
    if task_config.get("schema_version") == 2:
        from src.task_spec import TaskSpec
        spec = TaskSpec.from_mapping(task_config, task_id="validator/task")
        # TaskSession already ran initial actions. Budget only the semantic
        # review; never multiply action timeouts by the number of commands.
        review_timeout = agent_config.get("timeout_seconds", 1200)
        if type(review_timeout) is not int or review_timeout < 0:
            raise ValueError("Validator timeout_seconds must be a nonnegative integer")
        return (*(spec.action("baseline", a).timeout_s for a in ("compile", "correctness", "performance")), review_timeout)
    compile_timeout = _positive_timeout(
        task_config.get("compile_timeout"),
        _positive_timeout(agent_config.get("compile_timeout"), 300),
    )
    correctness_timeout = _positive_timeout(
        task_config.get("correctness_timeout"),
        _positive_timeout(agent_config.get("correctness_timeout"), 300),
    )
    performance_timeout = _positive_timeout(
        task_config.get("performance_timeout"),
        _positive_timeout(agent_config.get("performance_timeout"), 300),
    )
    configured_backend_timeout = agent_config.get("timeout_seconds", 1200)
    try:
        configured_backend_timeout = int(configured_backend_timeout)
    except (TypeError, ValueError):
        configured_backend_timeout = 1200
    if configured_backend_timeout <= 0:
        backend_timeout = 0
    else:
        command_counts = [
            max(1, len(commands)) if isinstance(commands, list) else 1
            for commands in (
                task_config.get("compile_command"),
                task_config.get("correctness_command"),
                task_config.get("performance_command"),
            )
        ]
        backend_timeout = max(
            configured_backend_timeout,
            compile_timeout * command_counts[0]
            + correctness_timeout * command_counts[1]
            + performance_timeout * command_counts[2]
            + 300,
        )
    return compile_timeout, correctness_timeout, performance_timeout, backend_timeout


def _resolve_backend_settings(
    eval_config: dict[str, Any], agent_config: dict[str, Any]
) -> tuple[str, str | None, str | None]:
    """Resolve validator backend settings with per-run overrides first."""

    run_agent = eval_config.get("agent")
    if not isinstance(run_agent, dict):
        run_agent = {}

    def _value(name: str, default: Any) -> Any:
        configured = run_agent.get(name)
        return default if configured in (None, "") else configured

    backend = str(_value("backend", agent_config.get("backend", "codex")))
    changed_backend = backend != agent_config.get("backend", "codex")
    model = _value("model", None if changed_backend else agent_config.get("model"))
    effort = _value("effort", None if changed_backend else agent_config.get("effort"))
    for name, value in (("model", model), ("effort", effort)):
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"Validator {name} must be a nonempty string")
    return backend, model, effort



def _expected_task_name(task_config_dir: str) -> str:
    path = Path(task_config_dir).resolve()
    parts = path.parts
    if "tasks" in parts:
        return Path(*parts[parts.index("tasks") + 1 : -1]).as_posix()
    return path.parent.name


@register_agent("task_validator")
def launch_agent(eval_config: dict[str, Any], task_config_dir: str, workspace: str) -> str:
    """Review task semantics after framework execution and finalize its draft.

    v2 context transport: eval_config['_task_validation_context'] or
    ARENA_VALIDATION_CONTEXT. The context is loaded exactly once before launch.
    The caller receives the generated request ID in
    eval_config['_task_validation_request_id'] for an optional final recheck with
    its own TaskSession memory snapshot. A failure still writes a complete FAIL.
    """
    logger = logging.getLogger(__name__)
    expected_task_name = eval_config.get("_task_id") or _expected_task_name(task_config_dir)
    trusted = None
    request_id = None
    is_v2 = False
    try:
        with Path(__file__).with_name("agent_config.yaml").open() as handle:
            agent_config = yaml.safe_load(handle) or {}
        with Path(task_config_dir).open() as handle:
            task_config = yaml.safe_load(handle)
        if not isinstance(task_config, dict):
            raise ValueError("config.yaml top level must be a mapping")
        is_v2 = task_config.get("schema_version") == 2
        run_agent = eval_config.get("agent", {})
        if not isinstance(run_agent, dict):
            raise ValueError("agent must be a mapping")
        config = dict(agent_config)
        for key in ("timeout_seconds", "python_path", "max_budget_usd"):
            if key in run_agent:
                config[key] = run_agent[key]
        backend, model, effort = _resolve_backend_settings(eval_config, agent_config)
        compile_timeout, correctness_timeout, performance_timeout, timeout = _resolve_validation_timeouts(task_config, config)
        python = config.get("python_path") or os.environ.get(PYTHON_ENV_VAR) or sys.executable
        eval_config.setdefault("agent", {}).update(
            python_path=python, compile_timeout=compile_timeout,
            correctness_timeout=correctness_timeout, performance_timeout=performance_timeout,
        )
        context_path = None
        context_file_hash = None
        if is_v2:
            request_id = uuid.uuid4().hex
            eval_config["_task_validation_request_id"] = request_id
            context_path = eval_config.get("_task_validation_context") or os.environ.get("ARENA_VALIDATION_CONTEXT")
            if not isinstance(context_path, str) or not context_path:
                raise ValueError("Schema-v2 validator requires a pre-launch framework task validation context")
            # Hash around loading as well as after execution, rejecting replacement
            # or a mutation during the capture window.
            context_file_hash = hashlib.sha256(Path(context_path).read_bytes()).hexdigest()
            trusted = load_task_evidence(context_path, task_id=expected_task_name,
                                         workspace=workspace, task_config=task_config)
            if hashlib.sha256(Path(context_path).read_bytes()).hexdigest() != context_file_hash:
                raise ValueError("Task validation context changed while being captured")
            eval_config["_task_validation_evidence_sha256"] = trusted.sha256
        else:
            gpu_check = subprocess.run(["rocm-smi", "--showid"], capture_output=True, text=True, timeout=10)
            if gpu_check.returncode != 0:
                raise RuntimeError("No AMD GPU detected; task validation requires compatible GPU execution")
        prompt = build_validation_prompt(task_config_dir, workspace, eval_config,
                                         trusted_task_evidence=trusted, validation_request_id=request_id)
        logger.info("Task validator backend=%s model=%s effort=%s timeout=%s task=%s",
                    backend, model, effort, timeout, expected_task_name)
        if backend == "codex":
            result = _launch_codex(prompt, workspace, timeout, logger, model=model, effort=effort)
        elif backend == "claude_code":
            result = _launch_claude_code(prompt, workspace, timeout, logger, model=model, effort=effort,
                                       max_budget_usd=config.get("max_budget_usd"))
        else:
            raise ValueError(f"Unsupported task_validator backend: {backend}")
        failures = []
        if result.timed_out:
            failures.append(f"Validator backend timed out after {timeout} seconds")
        if result.returncode != 0:
            failures.append(f"Validator backend exited with code {result.returncode}")
        if result.error:
            failures.append(result.error)
        if context_path:
            try:
                # The finalizer receives the original immutable object regardless
                # of what the backend did to this readable transport file.
                if hashlib.sha256(Path(context_path).read_bytes()).hexdigest() != context_file_hash:
                    failures.append("Task validation context changed during backend execution")
            except OSError:
                failures.append("Task validation context disappeared during backend execution")
        eval_config["_task_validation_backend_error"] = "; ".join(failures) or None
        report = finalize_report(workspace, expected_task_name=expected_task_name,
                                 framework_error=eval_config["_task_validation_backend_error"],
                                 trusted_task_evidence=trusted, validation_request_id=request_id,
                                 task_schema_version=2 if is_v2 else None)
        logger.info("Framework-finalized validation report: overall=%s", report["overall_status"])
        return result.output
    except Exception as exc:
        error = f"Validator operational failure: {type(exc).__name__}: {exc}"
        logger.error(error, exc_info=True)
        eval_config["_task_validation_backend_error"] = error
        finalize_report(workspace, expected_task_name=expected_task_name, framework_error=error,
                        trusted_task_evidence=trusted, validation_request_id=request_id,
                        task_schema_version=2 if is_v2 else None)
        return error
