# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import subprocess
from pathlib import Path
from typing import Protocol

from .config import BackendConfig


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
        binary = shutil.which("codex")
        if not binary:
            raise RuntimeError("codex CLI is required for quality_loop")
        workspace = workspace.resolve()
        no_gh_dir = workspace / ".quality_loop_no_gh"
        no_gh_dir.mkdir(exist_ok=True)
        env = os.environ.copy()
        for key in (
            "GH_TOKEN",
            "GITHUB_TOKEN",
            "SSH_AUTH_SOCK",
            "GIT_ASKPASS",
            "GIT_SSH_COMMAND",
        ):
            env.pop(key, None)
        env["GH_CONFIG_DIR"] = str(no_gh_dir)
        env["GIT_CONFIG_GLOBAL"] = os.devnull
        env["GIT_CONFIG_NOSYSTEM"] = "1"

        command = [
            binary,
            "exec",
            "--json",
            "--dangerously-bypass-approvals-and-sandbox",
            "--skip-git-repo-check",
            "--ephemeral",
            "-c",
            "features.memories=false",
            "--cd",
            str(workspace),
        ]
        if self.config.model:
            command.extend(["--model", self.config.model])
        if self.config.effort:
            command.extend(["-c", f'model_reasoning_effort={json.dumps(self.config.effort)}'])
        command.extend(["--", prompt])

        self.logger.info(
            "Starting Codex role=%s model=%s effort=%s workspace=%s",
            role,
            self.config.model or "<default>",
            self.config.effort,
            workspace,
        )
        with subprocess.Popen(
                command,
                cwd=workspace,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
        ) as process:
            try:
                stdout, stderr = process.communicate(timeout=self.config.timeout_seconds)
            except BaseException as exc:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.communicate()
                if isinstance(exc, subprocess.TimeoutExpired):
                    raise RuntimeError(
                        f"Codex role {role} timed out after {self.config.timeout_seconds}s"
                    ) from exc
                raise
            result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
        output = "\n".join(
            _format_event(line) for line in result.stdout.splitlines() if line.strip()
        )
        if result.returncode != 0:
            detail = (result.stderr or output).strip()
            raise RuntimeError(f"Codex role {role} failed ({result.returncode}): {detail[-4000:]}")
        completed = False
        for line in result.stdout.splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict) and event.get("type") == "turn.failed":
                raise RuntimeError(f"Codex role {role} reported a failed turn")
            if isinstance(event, dict) and event.get("type") == "turn.completed":
                completed = True
        if not completed:
            raise RuntimeError(f"Codex role {role} ended without a completed turn")
        if result.stderr.strip():
            self.logger.warning("Codex role=%s stderr: %s", role, result.stderr[-1000:])
        return output
