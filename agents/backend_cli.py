# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Run a model CLI backend (Codex or Claude Code) as one bounded process group.

Shared by the QA auditor's semantic review and the candidate verifier. The
backend's stream is logged; only terminal success events count as success.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import math
import os
import shutil
import signal
import subprocess
import threading
from typing import Any

from agents.prompt_input import prompt_input
from agents.stream_log import StreamLog


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


def run_backend(cmd: list[str], *, backend: str, workspace: str, timeout_seconds: int,
                 logger: logging.Logger, env: dict | None = None,
                 stdin=subprocess.DEVNULL, log_prefix: str = "backend-") -> BackendResult:
    if not shutil.which(cmd[0]):
        raise RuntimeError(f"Command {cmd[0]!r} not found; install/authenticate the model backend")
    # Only argv metadata is logged here; never interpolate shell commands.
    logger.info("Launching model backend %s in %s", backend, workspace)
    event_log = StreamLog(workspace, log_prefix + backend, logger, env=env)
    process = subprocess.Popen(cmd, stdin=stdin, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, errors="replace", cwd=workspace,
                               env=env, bufsize=1, start_new_session=True)
    stdout, stderr = [], []
    failed = threading.Event()
    completed = threading.Event()

    def read_stream(stream, output, is_stdout):
        try:
            with event_log.open("stdout" if is_stdout else "stderr") as record:
                for raw in stream:
                    line = record(raw)
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
                            logger.info("Backend %s event: %s", backend, kind)
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
    error = event_log.error
    if any(reader.is_alive() for reader in readers):
        # A child holding a pipe open is still this invocation's process. It must
        # not leak work into the next task even if the CLI parent already exited.
        _stop_process(process)
        for reader in readers:
            reader.join(timeout=5)
        error = "Backend stream did not close after exit"
    if failed.is_set():
        error = "Backend emitted a terminal failure event"
    elif not completed.is_set() and not timed_out:
        error = error or "Backend did not emit a successful terminal event"
    output = "\n".join(stdout)
    if stderr:
        output += "\n=== STDERR ===\n" + "\n".join(stderr)
    return BackendResult(output, process.returncode, timed_out, error)


def launch_codex(prompt: str, workspace: str, timeout_seconds: int, logger: logging.Logger,
                  model: str | None = None, effort: str | None = None) -> BackendResult:
    cmd = ["codex", "exec", "--json", "--dangerously-bypass-approvals-and-sandbox",
           "--skip-git-repo-check", "--ephemeral", "-c", "features.memories=false", "--cd", workspace]
    if model:
        cmd.extend(["--model", model])
    if effort:
        cmd.extend(["-c", f"model_reasoning_effort={json.dumps(effort)}"])
    cmd.extend(["--", "-"])
    with prompt_input(prompt) as stream:
        return run_backend(cmd, backend="codex", workspace=workspace,
                            timeout_seconds=timeout_seconds, logger=logger, stdin=stream)


def launch_claude_code(prompt: str, workspace: str, timeout_seconds: int, logger: logging.Logger,
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
        return run_backend(cmd, backend="claude_code", workspace=workspace, timeout_seconds=timeout_seconds,
                            logger=logger, env=env, stdin=stream)


def resolve_backend_settings(
    eval_config: dict[str, Any], agent_config: dict[str, Any]
) -> tuple[str, str | None, str | None]:
    """Resolve backend, model and effort with per-run overrides first."""

    run_agent = eval_config.get("agent")
    if not isinstance(run_agent, dict):
        run_agent = {}

    def _value(name: str, default: Any) -> Any:
        # Explicit null selects the backend's own default. In particular a
        # Codex run must not inherit a Claude model from the configured defaults.
        if name in ("model", "effort") and name in run_agent:
            return run_agent[name] or None
        configured = run_agent.get(name)
        return default if configured in (None, "") else configured

    backend = str(_value("backend", agent_config.get("backend", "codex")))
    changed_backend = backend != agent_config.get("backend", "codex")
    model = _value("model", None if changed_backend else agent_config.get("model"))
    effort = _value("effort", None if changed_backend else agent_config.get("effort"))
    for name, value in (("model", model), ("effort", effort)):
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"Backend {name} must be a nonempty string")
    return backend, model, effort
