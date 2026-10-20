# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import subprocess
import shutil
import logging
import math
import re
import threading
import time
import uuid
import os
import shlex
import sys
from pathlib import Path
from typing import Any
import yaml
from agents import register_agent
from agents.prompt_input import prompt_input
from agents.run_budget import append_run_budget
from src.module_registration import AgentType, load_prompt_builder
from src.process_control import stop_process_group as _stop_process
from src.prompts.agent_budget import iteration_budget_guidance, time_budget_guidance
from src.runtime_env import PYTHON_ENV_VAR, build_subprocess_env
from src.tools.agent_cost import report_agent_usage
from src.tools.agent_time_budget import (
    SessionOutcome, raise_for_sessions, resolve_budget_mode, run_sessions,
)

# Since 2.1.277 a headless resume restores the session's running cost, so
# result.total_cost_usd is a session total rather than this process's spend.
_SESSION_COST_SINCE = (2, 1, 277)


def _load_agent_config(eval_config: dict[str, Any]) -> dict[str, Any]:
    """Run settings override defaults; task configs never select a provider/model."""
    with Path(__file__).with_name("agent_config.yaml").open() as f:
        config = yaml.safe_load(f) or {}
    overrides = eval_config.get("agent", {})
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, dict):
        raise ValueError("agent must be a mapping")
    for key in (
        "model", "effort", "timeout_seconds", "max_iterations", "python_path",
        "max_budget_usd", "budget_mode",
    ):
        if key in overrides:
            config[key] = overrides[key]
    timeout = config.get("timeout_seconds")
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
        raise ValueError("agent.timeout_seconds must be a positive integer")
    for key in ("model", "effort"):
        value = config.get(key)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"agent.{key} must be a nonempty string or null")
    budget = config.get("max_budget_usd")
    if budget is not None and (
        isinstance(budget, bool) or not isinstance(budget, (int, float))
        or not math.isfinite(budget) or budget <= 0
    ):
        raise ValueError("agent.max_budget_usd must be a positive finite number or null")
    config["budget_mode"] = resolve_budget_mode(config)
    return config


def _build_command(
    agent_bin: str, prompt: str | None, config: dict[str, Any],
    session_id: str | None = None, resume: bool = False,
) -> list[str]:
    cmd = [
        agent_bin, "--print", "--verbose", "--output-format", "stream-json",
        "--include-partial-messages", "--permission-mode", "bypassPermissions",
    ]
    # Only time budgets resume a session, so only they persist one.
    if session_id is None:
        cmd.append("--no-session-persistence")
    else:
        cmd.extend(["--resume" if resume else "--session-id", session_id])
    for key, option in (
        ("model", "--model"), ("effort", "--effort"),
        ("max_budget_usd", "--max-budget-usd"),
    ):
        if config.get(key) is not None:
            cmd.extend([option, str(config[key])])
    cmd.extend(["--input-format", "text"] if prompt is None else ["--", prompt])
    return cmd


def _get_cli_version(agent_cmd: str) -> str:
    """Best-effort CLI version lookup for logging."""
    try:
        result = subprocess.run(
            [agent_cmd, "--version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except Exception:
        return "unknown"
    text = (result.stdout or result.stderr or "").strip()
    return text or "unknown"


def _cost_scope(version_text: str) -> str | None:
    """Scope of result.total_cost_usd for this CLI version; None if unknown."""
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", version_text or "")
    if not match:
        return None
    version = tuple(int(part) for part in match.groups())
    return "session" if version >= _SESSION_COST_SINCE else "invocation"


def _runtime_python_path(agent_config: dict[str, Any]) -> str:
    """Return the explicit or current framework Python for agent guidance."""
    configured = agent_config.get("python_path")
    if configured:
        return str(configured)
    return os.environ.get(PYTHON_ENV_VAR) or sys.executable


def integrate_agent_config(prompt: str, agent_config: dict[str, Any], python_path: str) -> str:
    """
    Integrate agent config into prompt.
    """
    if resolve_budget_mode(agent_config) == "time":
        prompt = prompt.rstrip() + time_budget_guidance()
    else:
        guidance = iteration_budget_guidance(agent_config.get("max_iterations"))
        if guidance:
            prompt = prompt.rstrip() + guidance
    prompt = append_run_budget(prompt, agent_config.get("timeout_seconds"))
    if python_path:
        prompt = prompt.rstrip() + f"\n\nUse this Python interpreter: `{python_path}`."
    return prompt


@register_agent("claude_code")
def launch_agent(eval_config: dict[str, Any], task_config_dir: str, workspace: str) -> str:
    """
    Launch Claude Code agent with real-time output streaming.

    Args:
        eval_config: Evaluator settings passed from main (includes task metadata like task_type)
        task_config_dir: Path to the task configuration used to build the prompt
        workspace: Workspace directory where the agent will run and read/write files

    Returns:
        str: Combined agent output (stdout plus stderr summary if present)
    """
    AGENT = "claude"
    agent_bin = shutil.which(AGENT)
    if not agent_bin:
        raise RuntimeError(
            f"Command '{AGENT}' not found. Please ensure Claude Code CLI is installed and in your PATH."
        )

    agent_config = _load_agent_config(eval_config)
    logger = logging.getLogger(__name__)

    prompt_builder = load_prompt_builder(AgentType.CLAUDE_CODE, logger)
    prompt = prompt_builder(task_config_dir, workspace, eval_config, logger)

    runtime_python = _runtime_python_path(agent_config)
    prompt = integrate_agent_config(prompt, agent_config, runtime_python)
    configured_model = agent_config.get("model")
    configured_effort = agent_config.get("effort")
    # IS_SANDBOX=1 allows skip-permissions even when invoked from a privileged user.
    # CLAUDE_CODE_DISABLE_AUTO_MEMORY=1 turns off the auto-memory feature (ON by
    # default in CLI >=2.1.59) so headless runs never read/write learned memory.
    process_env = build_subprocess_env(runtime_python)
    process_env.update(IS_SANDBOX="1", CLAUDE_CODE_DISABLE_AUTO_MEMORY="1")
    budget_mode = agent_config["budget_mode"]
    # A time budget resumes the session, so it needs a known, persisted id.
    session_id = str(uuid.uuid4()) if budget_mode == "time" else None
    cmd = _build_command(agent_bin, None, agent_config, session_id)
    cli_version = _get_cli_version(agent_bin)
    cost_scope = _cost_scope(cli_version)

    logger.info("Claude Code Preflight")
    logger.info(f"  binary: {agent_bin}")
    logger.info(f"  version: {cli_version}")
    logger.info(f"  workspace: {workspace}")
    logger.info(f"  python_path: {runtime_python}")
    logger.info(f"  model: {configured_model if configured_model else '<claude CLI default/config>'}")
    logger.info(f"  effort: {configured_effort if configured_effort else '<claude CLI default/config>'}")
    logger.info(f"  budget: {budget_mode} (timeout {agent_config.get('timeout_seconds')}s)")

    logger.info("Running command: %s <stdin prompt>", shlex.join(cmd))
    logger.info("=" * 80)
    logger.info("Agent Output (streaming):")
    logger.info("=" * 80)

    timeout_seconds = int(agent_config.get("timeout_seconds", 300))
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    def invoke(prompt_text: str, resume_id: str | None, invoke_timeout: float) -> SessionOutcome:
        started = time.monotonic()
        run_cmd = cmd if resume_id is None else _build_command(
            agent_bin, None, agent_config, resume_id, resume=True)
        if resume_id is not None:
            logger.info("Resuming command: %s <stdin prompt>", shlex.join(run_cmd))
        with prompt_input(prompt_text) as stream:
            process = subprocess.Popen(
                run_cmd, stdin=stream, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, cwd=workspace, bufsize=1, env=process_env, start_new_session=True,
            )
        failed_result = threading.Event()
        # Raw provider values from result events. One invocation can emit more
        # than one result: their usage covers disjoint parts and is summed, while
        # total_cost_usd is a running total, so the last value is kept.
        result_info: dict[str, Any] = {"usage": [], "cost_usd": None}

        def format_agent_event(data):
            """Convert Claude stream-json payloads into a readable single-line string."""
            if not isinstance(data, dict):
                return str(data)

            event_type = data.get("type")
            if event_type == "result" and (
                data.get("is_error") or str(data.get("subtype", "")).startswith("error")
            ):
                failed_result.set()
            if event_type == "result":
                if data.get("total_cost_usd") is not None:
                    result_info["cost_usd"] = data.get("total_cost_usd")
                if isinstance(data.get("usage"), dict):
                    result_info["usage"].append(data["usage"])
                if isinstance(data.get("session_id"), str):
                    result_info["session_id"] = data["session_id"]
            if event_type == "assistant":
                content = data.get("message", {}).get("content", [])
                texts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        texts.append(part.get("text", ""))
                text = " ".join(t.strip() for t in texts if t and t.strip())
                if not text:
                    return None
                return f"assistant: {text}"

            if event_type == "thinking":
                text = " ".join((data.get("text") or "").split())
                subtype = data.get("subtype")
                if not text:
                    return None
                return f"thinking[{subtype}] {text}" if subtype else f"thinking {text}"

            if event_type == "tool_call":
                subtype = data.get("subtype")
                call = data.get("tool_call") or {}
                call_name = next(iter(call.keys()), "unknown_tool")
                args = call.get(call_name, {}).get("args", {}) if isinstance(call, dict) else {}
                summary = []
                if isinstance(args, dict):
                    if "path" in args:
                        summary.append(f"path={args.get('path')}")
                    if "command" in args:
                        summary.append(f"cmd={args.get('command')}")
                details = " ".join(summary)
                return f"tool_call[{subtype}] {call_name} {details}".strip()

            if event_type == "user":
                message = data.get("message", {}).get("content", [])
                texts = []
                for part in message:
                    if isinstance(part, dict) and part.get("type") == "text":
                        texts.append(part.get("text", ""))
                text = " ".join(t.strip() for t in texts if t and t.strip())
                if not text:
                    return None
                text = " ".join(text.split())
                return f"user: {text[:160]}{'...' if len(text) > 160 else ''}"

            if event_type == "system":
                model = data.get("model")
                if isinstance(model, str):
                    result_info["model"] = model
                cwd = data.get("cwd")
                return f"system init model={model} cwd={cwd}"

            import json
            return json.dumps(data, ensure_ascii=False, separators=(",", ":"))

        def read_stream(stream, output_list, prefix, log_func):
            """Read from stream in a separate thread to avoid blocking."""
            import json
            import ast

            # Accumulate partial text per content_block index so we only log full sentences.
            text_buffers: dict[int, str] = {}

            def flush_buffer(idx: int):
                text = text_buffers.pop(idx, "").strip()
                if text:
                    line = f"assistant: {text}"
                    output_list.append(line)
                    log_func(f"{prefix} {line}")

            def handle_stream_event(data: dict):
                ev = data.get("event", {}) if isinstance(data, dict) else {}
                ev_type = ev.get("type")

                # Content blocks carry assistant text/tool info.
                if ev_type in ("content_block_start", "content_block_delta", "content_block_stop"):
                    idx = ev.get("index")
                    if idx is None:
                        return
                    block = ev.get("content_block") or {}
                    delta = ev.get("delta") or {}

                    # Start: set up buffer for text blocks.
                    if ev_type == "content_block_start":
                        if block.get("type") == "text":
                            text_buffers[idx] = ""
                        elif block.get("type") == "tool_use":
                            name = block.get("name")
                            inputs = block.get("input")
                            line = f"tool_use start {name} {inputs}"
                            output_list.append(line)
                            log_func(f"{prefix} {line}")
                    # Delta: append partial text.
                    elif ev_type == "content_block_delta":
                        if delta.get("type") == "text_delta":
                            text_buffers[idx] = text_buffers.get(idx, "") + delta.get("text", "")
                    # Stop: flush accumulated text.
                    elif ev_type == "content_block_stop":
                        flush_buffer(idx)
                    return

                # Skip noisy message envelope events.
                if ev_type in ("message_start", "message_delta", "message_stop"):
                    return

                # System init / other top-level events.
                formatted = format_agent_event(ev if ev else data)
                if formatted:
                    output_list.append(formatted)
                    log_func(f"{prefix} {formatted}")

            try:
                for line in iter(stream.readline, ''):
                    if not line:
                        break
                    raw_line = line.rstrip()
                    try:
                        data = json.loads(raw_line)
                    except json.JSONDecodeError:
                        try:
                            data = ast.literal_eval(raw_line)
                        except Exception:
                            data = None

                    if isinstance(data, dict) and data.get("type") == "stream_event":
                        handle_stream_event(data)
                        continue

                    formatted = format_agent_event(data) if data is not None else None
                    if formatted:
                        output_list.append(formatted)
                        log_func(f"{prefix} {formatted}")
                        continue

                    if raw_line.strip():
                        output_list.append(raw_line)
                        log_func(f"{prefix} {raw_line}")
            finally:
                stream.close()

        stdout_thread = threading.Thread(
            target=read_stream,
            args=(process.stdout, stdout_lines, "[AGENT]", logger.info),
            daemon=True
        )
        stderr_thread = threading.Thread(
            target=read_stream,
            args=(process.stderr, stderr_lines, "[AGENT STDERR]", logger.warning),
            daemon=True
        )

        stdout_thread.start()
        stderr_thread.start()

        timed_out = False
        try:
            process.wait(timeout=invoke_timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            logger.warning(f"Claude Code timed out after {invoke_timeout:.0f}s; terminating process")
            _stop_process(process)
        except BaseException:
            _stop_process(process)
            raise

        stdout_thread.join(timeout=1)
        stderr_thread.join(timeout=1)

        if stderr_lines:
            logger.warning("=" * 80)
            logger.warning(f"Agent STDERR captured {len(stderr_lines)} lines")
            logger.warning("=" * 80)

        logger.info("=" * 80)
        logger.info(f"Agent completed with exit code: {process.returncode}")
        logger.info("=" * 80)
        seconds = time.monotonic() - started
        invocation_session = session_id or result_info.get("session_id")
        report_agent_usage(
            cost_usd=result_info["cost_usd"], cost_scope=cost_scope,
            usage=result_info["usage"] or None, usage_scope="invocation",
            session_id=invocation_session, source="claude_code stream-json result",
            model=result_info.get("model") or configured_model, timed_out=timed_out, seconds=seconds,
        )
        return SessionOutcome(
            timed_out=timed_out, returncode=process.returncode, failed=failed_result.is_set(),
            session_id=invocation_session, seconds=seconds,
        )

    runs = run_sessions(
        invoke, mode=budget_mode, prompt=prompt, timeout_seconds=timeout_seconds, logger=logger,
    )

    output = "\n".join(stdout_lines)
    if stderr_lines:
        output += "\n=== STDERR ===\n" + "\n".join(stderr_lines)

    raise_for_sessions(runs, mode=budget_mode, agent_label="Claude Code",
                       timeout_seconds=timeout_seconds)
    return output
