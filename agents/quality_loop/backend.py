# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
from __future__ import annotations

import json
import logging
import os
import shutil
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
        if not shutil.which("codex"):
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
            "codex",
            "exec",
            "--json",
            "--dangerously-bypass-approvals-and-sandbox",
            "--skip-git-repo-check",
            "-c",
            "features.memories=false",
            "--cd",
            str(workspace),
        ]
        if self.config.model:
            command.extend(["--model", self.config.model])
        if self.config.effort:
            command.extend(["-c", f'model_reasoning_effort="{self.config.effort}"'])
        command.append(prompt)

        self.logger.info(
            "Starting Codex role=%s model=%s effort=%s workspace=%s",
            role,
            self.config.model or "<default>",
            self.config.effort,
            workspace,
        )
        try:
            result = subprocess.run(
                command,
                cwd=workspace,
                env=env,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Codex role {role} timed out after {self.config.timeout_seconds}s"
            ) from exc
        output = "\n".join(
            _format_event(line) for line in result.stdout.splitlines() if line.strip()
        )
        if result.returncode != 0:
            detail = (result.stderr or output).strip()
            raise RuntimeError(f"Codex role {role} failed ({result.returncode}): {detail[-4000:]}")
        if result.stderr.strip():
            self.logger.warning("Codex role=%s stderr: %s", role, result.stderr[-1000:])
        return output
