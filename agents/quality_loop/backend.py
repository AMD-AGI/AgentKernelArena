# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import selectors
import shutil
import signal
import subprocess
import tempfile
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from agents.prompt_input import prompt_input

from .config import BackendConfig


STDOUT_LIMIT = 8 * 1024 * 1024
STDERR_LIMIT = 1024 * 1024
EVENT_LINE_LIMIT = 1024 * 1024
CLEANUP_SECONDS = 5


class _StreamCapture:
    """Bound retained bytes and JSON buffering; hash/drain even after truncation."""

    def __init__(self, file, limit: int, *, events: bool = False):
        self.file, self.limit, self.events = file, limit, events
        self.observed = self.retained = 0
        self.digest = hashlib.sha256()
        self.tail = b""
        self.pending = b""
        self.discard_line = self.oversized_event = False
        self.eof = self.completed = False
        self.failed_event = None
        self.terminal_events: list[dict] = []
        self.terminal_count = 0

    def feed(self, data: bytes):
        self.observed += len(data)
        self.digest.update(data)
        kept = data[:max(0, self.limit - self.retained)]
        written = self.file.write(kept)
        self.retained += written
        if written != len(kept):
            raise OSError("Incomplete quality_loop process evidence write")
        self.tail = (self.tail + data)[-4000:]
        if not self.events:
            return
        parts = data.split(b"\n")
        for index, part in enumerate(parts):
            if not self.discard_line:
                if len(self.pending) + len(part) > EVENT_LINE_LIMIT:
                    self.oversized_event = self.discard_line = True
                    self.pending = b""
                else:
                    self.pending += part
            if index < len(parts) - 1:
                if not self.discard_line:
                    self._event(self.pending)
                self.pending = b""
                self.discard_line = False

    def _event(self, line: bytes):
        try:
            event = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            return
        if not isinstance(event, dict):
            return
        kind = event.get("type")
        if kind == "turn.completed":
            self.completed = True
        elif kind in ("turn.failed", "thread.failed", "error"):
            self.failed_event = kind
        else:
            return
        self.terminal_count += 1
        summary = {"type": kind}
        usage = event.get("usage")
        if isinstance(usage, dict):
            summary["usage"] = {
                key: value for key in ("input_tokens", "cached_input_tokens", "output_tokens")
                if type(value := usage.get(key)) is int and value >= 0
            }
        # Retain the latest terminal summaries even when raw stdout hit its cap.
        self.terminal_events = (self.terminal_events + [summary])[-16:]

    def finish(self):
        self.eof = True
        if self.events and self.pending and not self.discard_line:
            self._event(self.pending)
        self.pending = b""

    def summary(self):
        return {"file": Path(self.file.name).name, "limit_bytes": self.limit,
                "observed_bytes": self.observed, "retained_bytes": self.retained,
                "truncated": self.observed > self.retained, "retention": "prefix",
                "observed_sha256": self.digest.hexdigest(), "eof": self.eof}


def _write_status(directory: Path, status: dict):
    temporary = directory / "process.json.tmp"
    with temporary.open("w", encoding="utf-8") as stream:
        os.chmod(temporary, 0o600)
        json.dump(status, stream, indent=2, ensure_ascii=True)
        stream.write("\n")
    temporary.replace(directory / "process.json")


def _kill_group(process):
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _drain(process, captures: dict, deadline: float):
    with selectors.DefaultSelector() as selector:
        for pipe, capture in captures.items():
            if not capture.eof:
                selector.register(pipe, selectors.EVENT_READ, capture)
        exit_deadline = None
        while selector.get_map() or process.poll() is None:
            now = time.monotonic()
            if process.poll() is not None and exit_deadline is None:
                # A descendant retaining a pipe must not extend the role budget.
                exit_deadline = min(deadline, now + CLEANUP_SECONDS)
            remaining = (exit_deadline or deadline) - now
            if remaining <= 0:
                raise subprocess.TimeoutExpired(process.args, 0)
            for key, _ in selector.select(min(remaining, 0.1)):
                data = os.read(key.fd, 65536)
                if data:
                    key.data.feed(data)
                else:
                    key.data.finish()
                    selector.unregister(key.fileobj)


class AgentBackend(Protocol):
    def run(self, prompt: str, workspace: Path, *, role: str) -> str: ...


def _format_event(line: str) -> str:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return line
    if not isinstance(payload, dict):
        return line
    if payload.get("type") in {"item.completed", "item.updated"}:
        item = payload.get("item") or {}
        if isinstance(item, dict) and item.get("type") == "agent_message":
            return str(item.get("text") or "")
    if payload.get("type") in {"turn.failed", "error"}:
        return str(payload.get("error") or payload.get("message") or line)
    return line


class CodexBackend:
    """Role-scoped Codex runner with GitHub credentials removed from children."""

    def __init__(self, config: BackendConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def run(self, prompt: str, workspace: Path, *, role: str) -> str:
        workspace = workspace.resolve()
        if not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", role):
            raise ValueError("quality_loop role must be a short filename-safe identifier")
        # A sibling of the task workspace is outside every task-relative edit scope.
        # mkdtemp gives each invocation its own private, non-overwriting receipt.
        evidence = Path(tempfile.mkdtemp(prefix=f".quality_loop-{role}-", dir=workspace.parent))
        started = time.monotonic()
        status = {
            "schema_version": 1, "role": role, "workspace": str(workspace),
            "model": self.config.model, "effort": self.config.effort,
            "timeout_seconds": self.config.timeout_seconds,
            "started_at": datetime.now(timezone.utc).isoformat(), "status": "starting",
            "prompt": {"transport": "anonymous_stdin", "bytes": len(prompt.encode("utf-8")),
                       "sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()},
        }
        _write_status(evidence, status)
        self.logger.info("Codex role=%s process evidence=%s", role, evidence)
        process = None
        captures = {}
        output = ""
        with ExitStack() as stack:
            stdout = stderr = None
            try:
                for name, limit in (("stdout", STDOUT_LIMIT), ("stderr", STDERR_LIMIT)):
                    file = stack.enter_context((evidence / f"{name}.log").open("xb", buffering=0))
                    os.chmod(file.name, 0o600)
                    captures[name] = _StreamCapture(file, limit, events=name == "stdout")
                stdout, stderr = captures["stdout"], captures["stderr"]
                binary = shutil.which("codex")
                if not binary:
                    raise RuntimeError("codex CLI is required for quality_loop")
                no_gh_dir = workspace / ".quality_loop_no_gh"
                no_gh_dir.mkdir(exist_ok=True)
                env = os.environ.copy()
                for key in ("GH_TOKEN", "GITHUB_TOKEN", "SSH_AUTH_SOCK", "GIT_ASKPASS", "GIT_SSH_COMMAND"):
                    env.pop(key, None)
                env["GH_CONFIG_DIR"] = str(no_gh_dir)
                env["GIT_CONFIG_GLOBAL"] = os.devnull
                env["GIT_CONFIG_NOSYSTEM"] = "1"
                command = [
                    binary, "exec", "--json", "--dangerously-bypass-approvals-and-sandbox",
                    "--skip-git-repo-check", "--ephemeral", "-c", "features.memories=false",
                    "--cd", str(workspace),
                ]
                if self.config.model:
                    command.extend(["--model", self.config.model])
                if self.config.effort:
                    command.extend(["-c", f'model_reasoning_effort={json.dumps(self.config.effort)}'])
                command.extend(["--", "-"])
                # Only this constructed argv is recorded: never prompt text or env.
                status["argv"] = command
                stream = stack.enter_context(prompt_input(prompt))
                deadline = time.monotonic() + self.config.timeout_seconds
                process = subprocess.Popen(
                    command, cwd=workspace, env=env, stdin=stream,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
                )
                stack.callback(process.stdout.close)
                stack.callback(process.stderr.close)
                status.update(status="running", pid=process.pid)
                _write_status(evidence, status)
                pipes = {process.stdout: stdout, process.stderr: stderr}
                try:
                    _drain(process, pipes, deadline)
                except subprocess.TimeoutExpired as exc:
                    status["timed_out"] = True
                    _kill_group(process)
                    try:
                        _drain(process, pipes, time.monotonic() + CLEANUP_SECONDS)
                    except subprocess.TimeoutExpired:
                        pass  # Still-open pipes remain explicitly incomplete in evidence.
                    raise RuntimeError(
                        f"Codex role {role} timed out after {self.config.timeout_seconds}s"
                    ) from exc
                if process.returncode != 0:
                    detail = (stderr.tail or stdout.tail).decode("utf-8", errors="replace").strip()
                    raise RuntimeError(f"Codex role {role} failed ({process.returncode}): {detail}")
                if stdout.oversized_event:
                    raise RuntimeError(f"Codex role {role} exceeded the JSON event line limit")
                if stdout.failed_event:
                    raise RuntimeError(f"Codex role {role} reported a failed turn ({stdout.failed_event})")
                if not stdout.completed:
                    raise RuntimeError(f"Codex role {role} ended without a completed turn")
                retained = (evidence / "stdout.log").read_text(encoding="utf-8", errors="replace")
                output = "\n".join(_format_event(line) for line in retained.splitlines() if line.strip())
                status["status"] = "succeeded"
                if stderr.observed:
                    self.logger.warning("Codex role=%s emitted stderr; see %s", role, evidence / "stderr.log")
            except BaseException as exc:
                status["status"] = "timed_out" if status.get("timed_out") else "failed"
                # Exception messages/child env can contain credentials; record only type.
                status["exception_type"] = type(exc).__name__
                raise
            finally:
                if process is not None:
                    # Also clean up descendants after normal CLI exit. Never leave a
                    # process that can continue editing the task after the role returns.
                    _kill_group(process)
                    try:
                        process.wait(timeout=CLEANUP_SECONDS)
                    except subprocess.TimeoutExpired:
                        status["cleanup_incomplete"] = True
                        status["status"] = "failed"
                    status["returncode"] = process.returncode
                status.update(
                    finished_at=datetime.now(timezone.utc).isoformat(),
                    elapsed_seconds=time.monotonic() - started,
                    streams={name: capture.summary() for name, capture in captures.items()},
                )
                if stdout is not None:
                    status["events"] = {
                        "completed_turn": stdout.completed, "failed_event": stdout.failed_event,
                        "oversized_line": stdout.oversized_event,
                        "terminal_count": stdout.terminal_count,
                        "last_terminal_events": stdout.terminal_events,
                    }
                # Failure to persist evidence is an operational error, never success.
                _write_status(evidence, status)
        if status.get("cleanup_incomplete"):
            raise RuntimeError(f"Codex role {role} process cleanup incomplete; see {evidence}")
        return output
