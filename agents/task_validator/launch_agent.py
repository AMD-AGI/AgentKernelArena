# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import subprocess
import shutil
import logging
import os
import sys
import threading
import shlex
import json
from pathlib import Path
from typing import Any
import yaml
from agents import register_agent
from agents.task_validator.validation_prompt import build_validation_prompt
from src.runtime_env import PYTHON_ENV_VAR


def _launch_claude_code(
    prompt: str,
    workspace: str,
    timeout_seconds: int,
    logger: logging.Logger,
    model: str | None = None,
    effort: str | None = None,
) -> str:
    """Launch Claude Code CLI with the validation prompt."""
    AGENT = "claude"
    # --dangerously-skip-permissions is exactly equivalent to
    # `--permission-mode bypassPermissions`, so we only pass the latter.
    OPTIONS = (
        "--print "
        "--verbose "
        "--output-format stream-json "
        "--include-partial-messages "
        "--permission-mode bypassPermissions"
    )

    if not shutil.which(AGENT):
        raise RuntimeError(
            f"Command '{AGENT}' not found. Please ensure Claude Code CLI is installed and in your PATH."
        )

    dynamic_options = OPTIONS
    if model:
        dynamic_options += f" --model {shlex.quote(str(model))}"
    if effort:
        dynamic_options += f" --effort {shlex.quote(str(effort))}"

    quoted_prompt = shlex.quote(prompt)
    # CLAUDE_CODE_DISABLE_AUTO_MEMORY=1 turns off auto-memory (ON by default in
    # CLI >=2.1.59) so headless validation never reads/writes learned memory.
    cmd = f"IS_SANDBOX=1 CLAUDE_CODE_DISABLE_AUTO_MEMORY=1 {AGENT} {dynamic_options} {quoted_prompt}"

    logger.info(f"Validator Claude Code model: {model if model else '<claude CLI default/config>'}")
    logger.info(f"Validator Claude Code effort: {effort if effort else '<claude CLI default/config>'}")
    logger.info(f"Running command: {cmd[:200]}...")

    process = subprocess.Popen(
        cmd,
        shell=True,  # nosec B602 -- shell=True is required to launch agent process
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=workspace,
        bufsize=1
    )
    if process.stdin:
        process.stdin.close()

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    def read_stream(stream, output_list, prefix, log_func):
        """Read from stream in a separate thread to avoid blocking."""
        import json
        try:
            for line in iter(stream.readline, ''):
                if not line:
                    break
                raw_line = line.rstrip()
                if raw_line.strip():
                    output_list.append(raw_line)
                    # Log a condensed version to avoid flooding
                    try:
                        data = json.loads(raw_line)
                        event_type = data.get("type", "")
                        subtype = data.get("subtype", "")
                        # High-volume, low-signal events: per-token thinking counters
                        # and status pings flood the log (~2/3 of all lines). Keep them
                        # in output_list but do not log them.
                        if event_type == "system" and subtype in ("thinking_tokens", "status"):
                            continue
                        if event_type == "stream_event":
                            ev = data.get("event", {})
                            ev_type = ev.get("type", "")
                            # Streaming envelope + partial deltas carry no standalone
                            # signal (the full content arrives in the top-level
                            # assistant/user events), so skip them in the log.
                            if ev_type in (
                                "content_block_start", "content_block_delta", "content_block_stop",
                                "message_start", "message_delta", "message_stop",
                            ):
                                continue
                        log_func(f"{prefix} {raw_line[:200]}")
                    except (json.JSONDecodeError, AttributeError):
                        log_func(f"{prefix} {raw_line[:200]}")
        finally:
            stream.close()

    stdout_thread = threading.Thread(
        target=read_stream,
        args=(process.stdout, stdout_lines, "[VALIDATOR]", logger.info),
        daemon=True
    )
    stderr_thread = threading.Thread(
        target=read_stream,
        args=(process.stderr, stderr_lines, "[VALIDATOR STDERR]", logger.warning),
        daemon=True
    )

    stdout_thread.start()
    stderr_thread.start()

    try:
        if timeout_seconds > 0:
            process.wait(timeout=timeout_seconds)
        else:
            process.wait()
    except subprocess.TimeoutExpired:
        logger.warning(f"Validator timed out after {timeout_seconds}s; terminating process")
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("Force killing validator process")
            process.kill()

    stdout_thread.join(timeout=5)
    stderr_thread.join(timeout=5)

    if stderr_lines:
        logger.warning(f"Validator STDERR captured {len(stderr_lines)} lines")

    logger.info(f"Validator completed with exit code: {process.returncode}")

    output = "\n".join(stdout_lines)
    if stderr_lines:
        output += "\n=== STDERR ===\n" + "\n".join(stderr_lines)

    return output


def _launch_codex(
    prompt: str,
    workspace: str,
    timeout_seconds: int,
    logger: logging.Logger,
    model: str | None = None,
    effort: str | None = None,
) -> str:
    """Launch Codex CLI in non-interactive mode for task validation."""
    AGENT = "codex"

    if not shutil.which(AGENT):
        raise RuntimeError(
            f"Command '{AGENT}' not found. Please ensure Codex CLI is installed and in your PATH."
        )

    # Highest privilege mode: bypass sandbox and approval prompts.
    cmd = [
        AGENT,
        "exec",
        "--json",
        "--dangerously-bypass-approvals-and-sandbox",
        "--skip-git-repo-check",
        # Disable cross-session "Memories" (off by default, pinned for safety).
        "-c",
        "features.memories=false",
        "--cd",
        workspace,
    ]
    if model:
        cmd.extend(["--model", str(model)])
    if effort:
        cmd.extend(["-c", f'model_reasoning_effort="{effort}"'])
    cmd.append(prompt)

    logger.info(f"Validator Codex model: {model if model else '<codex CLI default/config>'}")
    logger.info(f"Validator Codex effort: {effort if effort else '<codex config default>'}")
    logger.info(f"Running command: {' '.join(shlex.quote(p) for p in cmd[:8])} ...")

    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=workspace,
        bufsize=1,
    )
    if process.stdin:
        process.stdin.close()

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    def _format_codex_event(raw_line: str) -> str:
        """Format Codex JSONL events. Modern `codex exec --json` uses an
        item-based envelope ({"type":"item.completed","item":{"type":
        "agent_message","text":...}}); older flat/`msg` shapes are fallbacks."""
        try:
            data = json.loads(raw_line)
        except json.JSONDecodeError:
            return raw_line

        if not isinstance(data, dict):
            return raw_line

        ev_type = data.get("type", "")

        # Current item-based envelope.
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
                    return f"mcp_tool[{item.get('status', '')}] {item.get('server', '')}.{item.get('tool', '')}".strip()
                elif item_type == "file_change":
                    return f"file_change[{item.get('status', '')}]".strip()
                elif item_type == "error":
                    return f"error: {item.get('message', raw_line)}"
            return raw_line

        if ev_type == "turn.completed":
            usage = data.get("usage")
            if isinstance(usage, dict):
                return f"turn.completed usage in={usage.get('input_tokens')} out={usage.get('output_tokens')}"
            return raw_line

        if ev_type in {"turn.failed", "error"}:
            err = data.get("error") or data.get("message")
            if isinstance(err, dict):
                err = err.get("message", err)
            return f"{ev_type}: {err}" if err else raw_line

        if ev_type in {"thread.started", "turn.started"}:
            return raw_line

        # Legacy fallbacks (older Codex binaries).
        msg = data.get("msg")
        if isinstance(msg, dict) and msg.get("type") in {"agent_message", "assistant_message"}:
            text = msg.get("message") or msg.get("text")
            if isinstance(text, str) and text.strip():
                return f"assistant: {text.strip()}"
        if ev_type in {"assistant_message", "assistant"}:
            msg = data.get("message", {})
            if isinstance(msg, dict):
                content = msg.get("content")
                if isinstance(content, str) and content.strip():
                    return f"assistant: {content.strip()}"
            text = data.get("text")
            if isinstance(text, str) and text.strip():
                return f"assistant: {text.strip()}"
        if "text" in data and isinstance(data["text"], str) and data["text"].strip():
            return data["text"].strip()
        return raw_line

    def read_stream(stream, output_list, prefix, log_func):
        try:
            for line in iter(stream.readline, ""):
                if not line:
                    break
                raw_line = line.rstrip()
                if raw_line.strip():
                    formatted = _format_codex_event(raw_line)
                    output_list.append(formatted)
                    log_func(f"{prefix} {formatted[:240]}")
        finally:
            stream.close()

    stdout_thread = threading.Thread(
        target=read_stream,
        args=(process.stdout, stdout_lines, "[VALIDATOR]", logger.info),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=read_stream,
        args=(process.stderr, stderr_lines, "[VALIDATOR STDERR]", logger.warning),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()

    # timeout_seconds <= 0 means "wait until completion".
    try:
        if timeout_seconds > 0:
            process.wait(timeout=timeout_seconds)
        else:
            process.wait()
    except subprocess.TimeoutExpired:
        logger.warning(f"Validator timed out after {timeout_seconds}s; terminating process")
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("Force killing validator process")
            process.kill()

    stdout_thread.join(timeout=5)
    stderr_thread.join(timeout=5)

    if stderr_lines:
        logger.warning(f"Validator STDERR captured {len(stderr_lines)} lines")
    logger.info(f"Validator completed with exit code: {process.returncode}")

    output = "\n".join(stdout_lines)
    if stderr_lines:
        output += "\n=== STDERR ===\n" + "\n".join(stderr_lines)
    return output


@register_agent("task_validator")
def launch_agent(eval_config: dict[str, Any], task_config_dir: str, workspace: str) -> str:
    """
    Launch the task validation agent.

    This agent validates that a task is correctly configured and self-contained.
    It does NOT optimize kernels. Instead, it runs a series of checks and produces
    a validation_report.yaml in the workspace.

    Args:
        eval_config: Evaluator settings passed from main
        task_config_dir: Path to the task configuration directory's config.yaml
        workspace: Workspace directory where the agent will run

    Returns:
        str: Combined agent output
    """
    logger = logging.getLogger(__name__)

    # Load agent config
    config_path = Path(__file__).with_name("agent_config.yaml")
    with config_path.open("r") as f:
        agent_config = yaml.safe_load(f) or {}

    backend = agent_config.get("backend", "claude_code")
    timeout_seconds = int(agent_config.get("timeout_seconds", 600))
    configured_model = agent_config.get("model")
    configured_effort = agent_config.get("effort")
    # Resolve interpreter: explicit config -> framework-detected (set by main.py)
    # -> this process's interpreter. Avoids hardcoding a path that may not exist
    # inside the Docker container.
    python_path = (
        agent_config.get("python_path")
        or os.environ.get(PYTHON_ENV_VAR)
        or sys.executable
    )

    # Inject agent_config values into eval_config for the prompt builder
    agent_section = eval_config.setdefault("agent", {})
    agent_section["python_path"] = python_path
    agent_section["compile_timeout"] = int(agent_config.get("compile_timeout", 300))
    agent_section["correctness_timeout"] = int(agent_config.get("correctness_timeout", 300))
    agent_section["performance_timeout"] = int(agent_config.get("performance_timeout", 300))

    # GPU availability check — validation tasks require a GPU to run compile/correctness/performance
    try:
        gpu_check = subprocess.run(
            ["rocm-smi", "--showid"],
            capture_output=True, text=True, timeout=10
        )
        if gpu_check.returncode != 0:
            raise RuntimeError(
                "No AMD GPU detected. `rocm-smi --showid` failed. "
                "Task validation requires a GPU to run compile, correctness, and performance checks."
            )
    except FileNotFoundError:
        raise RuntimeError(
            "rocm-smi not found. ROCm toolkit is required for task validation. "
            "Please ensure ROCm is installed and rocm-smi is in your PATH."
        )

    logger.info(f"Task Validator: backend={backend}, timeout={timeout_seconds}s")
    logger.info(f"Task Validator model: {configured_model if configured_model else '<backend default/config>'}")
    logger.info(f"Task Validator effort: {configured_effort if configured_effort else '<backend default/config>'}")
    logger.info(f"Task config: {task_config_dir}")
    logger.info(f"Workspace: {workspace}")

    # Build validation prompt (custom, not using shared prompt_builder)
    prompt = build_validation_prompt(task_config_dir, workspace, eval_config)
    logger.info(f"Validation prompt built, length: {len(prompt)} characters")

    # Launch the chosen backend
    if backend == "claude_code":
        output = _launch_claude_code(
            prompt,
            workspace,
            timeout_seconds,
            logger,
            model=configured_model,
            effort=configured_effort,
        )
    elif backend == "codex":
        output = _launch_codex(
            prompt,
            workspace,
            timeout_seconds,
            logger,
            model=configured_model,
            effort=configured_effort,
        )
    elif backend == "cursor":
        # Placeholder for Cursor backend
        raise NotImplementedError("Cursor backend not yet implemented for task_validator")
    else:
        raise ValueError(f"Unknown backend: {backend}. Supported: claude_code, codex")

    # Verify that validation_report.yaml was generated
    report_path = Path(workspace) / "validation_report.yaml"
    if report_path.exists():
        logger.info(f"Validation report generated: {report_path}")
    else:
        logger.warning(f"WARNING: validation_report.yaml was NOT generated in {workspace}")

    return output
