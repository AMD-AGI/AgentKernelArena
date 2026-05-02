# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import json
import logging
import shlex
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any

import yaml

from agents import register_agent
from src.module_registration import AgentType, load_prompt_builder
from src.runtime_env import build_subprocess_env


def integrate_agent_config(prompt: str, agent_config: dict[str, Any]) -> str:
    """Append agent-specific guidance to the prompt."""
    max_iters = agent_config.get("max_iterations")
    if max_iters is not None:
        prompt = prompt.rstrip() + f"\n\nFor this optimization, you must iterate up to {max_iters} versions."
    python_path = agent_config.get("python_path")
    if python_path:
        prompt = prompt.rstrip() + (
            f"\n\nUse this Python interpreter: `{python_path}`. "
            f"When running pytest, use `{python_path} -m pytest` instead of bare `pytest`."
        )
    return prompt


def _format_codex_event(raw_line: str) -> str:
    """Convert Codex JSON lines into readable log lines."""
    try:
        data = json.loads(raw_line)
    except json.JSONDecodeError:
        return raw_line

    if not isinstance(data, dict):
        return raw_line

    ev_type = data.get("type", "")

    if ev_type in {"assistant_message", "assistant"}:
        message = data.get("message", {})
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return f"assistant: {content.strip()}"
        text = data.get("text")
        if isinstance(text, str) and text.strip():
            return f"assistant: {text.strip()}"

    if ev_type in {"tool_call", "tool_result"}:
        return raw_line

    if ev_type in {"error", "warning"}:
        return raw_line

    text = data.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()

    return raw_line


def _get_codex_version(agent_cmd: str, env: dict[str, str] | None = None) -> str:
    """Best-effort Codex CLI version lookup for logging."""
    try:
        result = subprocess.run(
            [agent_cmd, "--version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
            env=env,
        )
    except Exception:
        return "unknown"

    text = (result.stdout or result.stderr or "").strip()
    return text or "unknown"


@register_agent("codex")
def launch_agent(eval_config: dict[str, Any], task_config_dir: str, workspace: str) -> str:
    """
    Launch Codex CLI in non-interactive mode with streaming output capture.

    Args:
        eval_config: Evaluator settings passed from main
        task_config_dir: Path to the task configuration used to build the prompt
        workspace: Workspace directory where the agent runs

    Returns:
        Combined stdout/stderr output captured from Codex CLI.
    """
    AGENT = "codex"
    codex_bin = shutil.which(AGENT)
    if not codex_bin:
        raise RuntimeError(
            f"Command '{AGENT}' not found. Please ensure Codex CLI is installed and in your PATH."
        )

    config_path = Path(__file__).with_name("agent_config.yaml")
    with config_path.open("r") as f:
        agent_config = yaml.safe_load(f) or {}

    logger = logging.getLogger(__name__)
    prompt_builder = load_prompt_builder(AgentType.CODEX, logger)
    prompt = prompt_builder(task_config_dir, workspace, eval_config, logger)
    prompt = integrate_agent_config(prompt, agent_config)
    configured_model = agent_config.get("model")
    process_env = build_subprocess_env(agent_config.get("python_path"))

    cmd = [
        AGENT,
        "exec",
        "--json",
        "--dangerously-bypass-approvals-and-sandbox",
        "--skip-git-repo-check",
        "--cd",
        workspace,
    ]
    if configured_model:
        cmd.extend(["--model", str(configured_model)])
    cmd.append(prompt)

    logger.info("Codex Preflight")
    logger.info(f"  codex_binary: {codex_bin}")
    logger.info(f"  codex_version: {_get_codex_version(AGENT, process_env)}")
    logger.info(f"  workspace: {workspace}")
    logger.info(f"  python_path: {process_env.get('AGENT_KERNEL_ARENA_PYTHON', '<unset>')}")
    if configured_model:
        logger.info(f"  model: {configured_model} (explicit via agents/codex/agent_config.yaml)")
    else:
        logger.info("  model: <codex CLI default/config> (not explicitly set)")
    logger.info(f"Running command: {' '.join(shlex.quote(p) for p in cmd[:8])} ...")
    logger.info("=" * 80)
    logger.info("Agent Output (streaming):")
    logger.info("=" * 80)

    timeout_seconds = int(agent_config.get("timeout_seconds", 600))

    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=workspace,
        bufsize=1,
        env=process_env,
    )
    if process.stdin:
        process.stdin.close()

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    def read_stream(stream, output_list, prefix, log_func):
        try:
            for line in iter(stream.readline, ""):
                if not line:
                    break
                raw_line = line.rstrip()
                if not raw_line.strip():
                    continue
                formatted = _format_codex_event(raw_line)
                output_list.append(formatted)
                log_func(f"{prefix} {formatted[:240]}")
        finally:
            stream.close()

    stdout_thread = threading.Thread(
        target=read_stream,
        args=(process.stdout, stdout_lines, "[AGENT]", logger.info),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=read_stream,
        args=(process.stderr, stderr_lines, "[AGENT STDERR]", logger.warning),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()

    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        logger.warning(f"Codex agent timed out after {timeout_seconds}s; terminating process")
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("Force killing Codex agent process")
            process.kill()

    stdout_thread.join(timeout=5)
    stderr_thread.join(timeout=5)

    if stderr_lines:
        logger.warning("=" * 80)
        logger.warning(f"Agent STDERR captured {len(stderr_lines)} lines")
        logger.warning("=" * 80)

    logger.info("=" * 80)
    logger.info(f"Agent completed with exit code: {process.returncode}")
    logger.info("=" * 80)

    output = "\n".join(stdout_lines)
    if stderr_lines:
        output += "\n=== STDERR ===\n" + "\n".join(stderr_lines)
    return output
