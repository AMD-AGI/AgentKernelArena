# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import json
import logging
import os
import shlex
import shutil
import signal
import subprocess
import threading
from pathlib import Path
from typing import Any

import yaml

from agents import register_agent
from agents.prompt_input import prompt_input
from src.module_registration import AgentType, load_prompt_builder
from src.runtime_env import build_subprocess_env


def _load_agent_config(eval_config: dict[str, Any]) -> dict[str, Any]:
    """Resolve defaults plus run-level settings, independently of task files."""
    with Path(__file__).with_name("agent_config.yaml").open() as f:
        config = yaml.safe_load(f) or {}
    overrides = eval_config.get("agent", {})
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, dict):
        raise ValueError("agent must be a mapping")
    for key in ("model", "effort", "timeout_seconds", "max_iterations", "python_path"):
        if key in overrides:
            config[key] = overrides[key]
    timeout = config.get("timeout_seconds")
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
        raise ValueError("agent.timeout_seconds must be a positive integer")
    for key in ("model", "effort"):
        value = config.get(key)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"agent.{key} must be a nonempty string or null")
    return config


def _build_command(
    agent_bin: str, workspace: str, prompt: str | None, config: dict[str, Any]
) -> list[str]:
    """Build literal argv; None selects stdin prompt transport without a shell."""
    cmd = [
        agent_bin, "exec", "--json",
        "--dangerously-bypass-approvals-and-sandbox", "--skip-git-repo-check",
        "--ephemeral", "-c", "features.memories=false", "--cd", workspace,
    ]
    if config.get("model"):
        cmd.extend(["--model", config["model"]])
    if config.get("effort"):
        cmd.extend(["-c", f'model_reasoning_effort={json.dumps(config["effort"])}'])
    cmd.extend(["--", "-" if prompt is None else prompt])
    return cmd


def _stop_process(process: subprocess.Popen) -> None:
    """Terminate the invocation's process group, including compiler/tool children."""
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
    process.wait()


def integrate_agent_config(
    prompt: str,
    agent_config: dict[str, Any],
    python_path: str | None,
) -> str:
    """Append agent-specific guidance to the prompt."""
    max_iters = agent_config.get("max_iterations")
    if max_iters is not None:
        prompt = prompt.rstrip() + f"\n\nFor this optimization, you must iterate up to {max_iters} versions."
    if python_path:
        prompt = prompt.rstrip() + (
            f"\n\nUse this Python interpreter: `{python_path}`. "
            f"When running pytest, use `{python_path} -m pytest` instead of bare `pytest`."
        )
    return prompt


def _format_codex_event(raw_line: str) -> str:
    """Convert Codex JSONL events into readable log lines.

    Modern `codex exec --json` (>=0.x item-based stream) emits a thread/turn/item
    envelope, e.g.::

        {"type":"item.completed","item":{"type":"agent_message","text":"..."}}
        {"type":"item.started","item":{"type":"command_execution","command":"...","status":"in_progress"}}
        {"type":"turn.completed","usage":{...}}

    The assistant's final answer lives at ``item.text`` of an ``item.completed``
    event whose ``item.type == "agent_message"``. Older binaries used flat
    ``assistant_message``/``assistant`` events or a nested ``msg`` envelope; those
    are kept as fallbacks so logs stay readable across Codex versions.
    """
    try:
        data = json.loads(raw_line)
    except json.JSONDecodeError:
        return raw_line

    if not isinstance(data, dict):
        return raw_line

    ev_type = data.get("type", "")

    # --- Current item-based envelope -------------------------------------
    if ev_type in {"item.started", "item.completed", "item.updated"}:
        item = data.get("item") or {}
        if isinstance(item, dict):
            item_type = item.get("type", "")
            if item_type == "agent_message":
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    return f"assistant: {text.strip()}"
            elif item_type == "reasoning":
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    return f"reasoning: {text.strip()}"
            elif item_type == "command_execution":
                command = item.get("command", "")
                status = item.get("status", "")
                exit_code = item.get("exit_code")
                tail = f" exit={exit_code}" if exit_code is not None else ""
                return f"command[{status}] {command}{tail}".strip()
            elif item_type == "mcp_tool_call":
                server = item.get("server", "")
                tool = item.get("tool", "")
                status = item.get("status", "")
                return f"mcp_tool[{status}] {server}.{tool}".strip()
            elif item_type == "file_change":
                return f"file_change[{item.get('status', '')}]".strip()
            elif item_type == "error":
                return f"error: {item.get('message', raw_line)}"
        return raw_line

    if ev_type == "turn.completed":
        usage = data.get("usage")
        if isinstance(usage, dict):
            return (
                "turn.completed usage "
                f"in={usage.get('input_tokens')} out={usage.get('output_tokens')}"
            )
        return raw_line

    if ev_type in {"turn.failed", "error"}:
        err = data.get("error") or data.get("message")
        if isinstance(err, dict):
            err = err.get("message", err)
        return f"{ev_type}: {err}" if err else raw_line

    if ev_type in {"thread.started", "turn.started"}:
        return raw_line

    # --- Legacy fallbacks (older Codex binaries) -------------------------
    # Nested `msg` envelope: {"msg":{"type":"agent_message","message":"..."}}
    msg = data.get("msg")
    if isinstance(msg, dict) and msg.get("type") in {"agent_message", "assistant_message"}:
        text = msg.get("message") or msg.get("text")
        if isinstance(text, str) and text.strip():
            return f"assistant: {text.strip()}"

    # Oldest flat events.
    if ev_type in {"assistant_message", "assistant"}:
        message = data.get("message", {})
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return f"assistant: {content.strip()}"
        text = data.get("text")
        if isinstance(text, str) and text.strip():
            return f"assistant: {text.strip()}"

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

    agent_config = _load_agent_config(eval_config)

    logger = logging.getLogger(__name__)
    process_env = build_subprocess_env(agent_config.get("python_path"))
    prompt_builder = load_prompt_builder(AgentType.CODEX, logger)
    prompt = prompt_builder(task_config_dir, workspace, eval_config, logger)
    prompt = integrate_agent_config(
        prompt,
        agent_config,
        process_env.get("AGENT_KERNEL_ARENA_PYTHON"),
    )
    configured_model = agent_config.get("model")
    configured_effort = agent_config.get("effort")

    cmd = _build_command(codex_bin, workspace, None, agent_config)

    logger.info("Codex Preflight")
    logger.info(f"  codex_binary: {codex_bin}")
    logger.info(f"  codex_version: {_get_codex_version(codex_bin, process_env)}")
    logger.info(f"  workspace: {workspace}")
    logger.info(f"  python_path: {process_env.get('AGENT_KERNEL_ARENA_PYTHON', '<unset>')}")
    if configured_model:
        logger.info(f"  model: {configured_model} (resolved run setting)")
    else:
        logger.info("  model: <codex CLI default/config> (not explicitly set)")
    logger.info(f"  effort: {configured_effort if configured_effort else '<codex config default>'} (model_reasoning_effort)")
    logger.info("Running command: %s <stdin prompt>", shlex.join(cmd))
    logger.info("=" * 80)
    logger.info("Agent Output (streaming):")
    logger.info("=" * 80)

    timeout_seconds = int(agent_config.get("timeout_seconds", 600))

    with prompt_input(prompt) as stream:
        process = subprocess.Popen(
            cmd, stdin=stream, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, cwd=workspace, bufsize=1, env=process_env, start_new_session=True,
        )

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    failed_turn = threading.Event()

    def read_stream(stream, output_list, prefix, log_func):
        try:
            for line in iter(stream.readline, ""):
                if not line:
                    break
                raw_line = line.rstrip()
                if not raw_line.strip():
                    continue
                formatted = _format_codex_event(raw_line)
                if formatted.startswith("turn.failed:"):
                    failed_turn.set()
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

    timed_out = False
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        logger.warning(f"Codex agent timed out after {timeout_seconds}s; terminating process")
        _stop_process(process)
    except BaseException:
        _stop_process(process)
        raise

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
    if timed_out:
        raise TimeoutError(f"Codex timed out after {timeout_seconds}s; see agent logs")
    if process.returncode != 0:
        raise RuntimeError(f"Codex exited with code {process.returncode}; see agent logs")
    if failed_turn.is_set():
        raise RuntimeError("Codex reported a failed turn; see agent logs")
    return output
