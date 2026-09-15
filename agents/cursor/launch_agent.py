# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import subprocess
import shutil
import logging
import threading
import os
import shlex
import signal
import sys
from pathlib import Path
from typing import Any
import yaml
from agents import register_agent
from src.module_registration import AgentType, load_prompt_builder
from src.runtime_env import PYTHON_ENV_VAR, build_subprocess_env


def _load_agent_config(eval_config: dict[str, Any]) -> dict[str, Any]:
    with Path(__file__).with_name("agent_config.yaml").open() as f:
        config = yaml.safe_load(f) or {}
    overrides = eval_config.get("agent", {})
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, dict):
        raise ValueError("agent must be a mapping")
    for key in ("model", "timeout_seconds", "max_iterations", "python_path", "effort"):
        if key in overrides:
            config[key] = overrides[key]
    timeout = config.get("timeout_seconds")
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
        raise ValueError("agent.timeout_seconds must be a positive integer")
    model = config.get("model")
    if model is not None and (not isinstance(model, str) or not model.strip()):
        raise ValueError("agent.model must be a nonempty string or null")
    if config.get("effort") is not None:
        raise ValueError(
            "Cursor has no standalone effort option; select a model variant from "
            "`cursor-agent models` or use a supported parameterized model ID"
        )
    return config


def _build_command(
    agent_bin: str, workspace: str, prompt: str, config: dict[str, Any]
) -> list[str]:
    cmd = [
        agent_bin, "--force", "--print", "--output-format", "stream-json",
        "--stream-partial-output", "--trust", "--workspace", workspace,
    ]
    if config.get("model"):
        cmd.extend(["--model", config["model"]])
    cmd.extend(["--", prompt])
    return cmd


def _stop_process(process: subprocess.Popen) -> None:
    """Terminate only this invocation and its tool children."""
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


def _runtime_python_path(agent_config: dict[str, Any]) -> str:
    """Return the explicit or current framework Python for agent guidance."""
    configured = agent_config.get("python_path")
    if configured:
        return str(configured)
    return os.environ.get(PYTHON_ENV_VAR) or sys.executable


def integrate_agent_config(prompt, agent_config: dict[str, Any], python_path: str) -> str:
    """
    Integrate agent config into prompt.
    """
    max_iters = agent_config.get("max_iterations")
    if max_iters is not None:
        prompt = prompt.rstrip() + f"\n\nFor this optimization, you must iterate up to {max_iters} versions."
    if python_path:
        prompt = prompt.rstrip() + f"\n\nUse this Python interpreter: `{python_path}`."
    return prompt

@register_agent("cursor")
def launch_agent(eval_config: dict[str, Any], task_config_dir: str, workspace: str) -> str:
    """
    Launch cursor agent with real-time output streaming.

    Args:
        eval_config: Evaluator settings passed from main (includes task metadata like task_type)
        task_config_dir: Path to the task configuration used to build the prompt
        workspace: Workspace directory where the agent will run and read/write files

    Returns:
        str: Combined agent output (stdout plus stderr summary if present)
    """
    AGENT = "cursor-agent"
    agent_config = _load_agent_config(eval_config)
    logger = logging.getLogger(__name__)

    # Check if the command exists
    agent_bin = shutil.which(AGENT)
    if not agent_bin:
        raise RuntimeError(
            f"Command '{AGENT}' not found. Please ensure cursor-agent is installed and in your PATH."
        )
    
    prompt_builder = load_prompt_builder(AgentType.CURSOR, logger)
    prompt = prompt_builder(task_config_dir, workspace, eval_config, logger)

    runtime_python = _runtime_python_path(agent_config)
    prompt = integrate_agent_config(prompt, agent_config, runtime_python)
    configured_model = agent_config.get("model")
    process_env = build_subprocess_env(runtime_python)
    cmd = _build_command(agent_bin, workspace, prompt, agent_config)

    logger.info("Cursor Agent Preflight")
    logger.info(f"  binary: {agent_bin}")
    logger.info(f"  version: {_get_cli_version(agent_bin)}")
    logger.info(f"  workspace: {workspace}")
    logger.info(f"  python_path: {runtime_python}")
    logger.info(f"  model: {configured_model if configured_model else '<cursor CLI default/config>'}")
    logger.info("  effort: <encoded in supported model variants or parameterized model IDs>")
    logger.info("Running command: %s <prompt>", shlex.join(cmd[:-1]))
    logger.info("=" * 80)
    logger.info("Agent Output (streaming):")
    logger.info("=" * 80)

    # Give the agent a hard stop to avoid blocking downstream tasks if it
    # keeps waiting for interactive input after finishing its work.
    timeout_seconds = int(agent_config.get("timeout_seconds", 300))

    # Use Popen for real-time output streaming with interactive input support
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,  # Keep stdin closed so the agent exits when done
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=workspace,
        bufsize=1,
        env=process_env,
        start_new_session=True,
    )

    # Close stdin immediately; leaving it attached keeps the agent alive waiting
    # for more user messages even after it reports completion.
    if process.stdin:
        process.stdin.close()

    # Collect output while streaming
    stdout_lines = []
    stderr_lines = []
    failed_result = threading.Event()

    def format_agent_event(data):
        """Convert cursor stream-json payloads into a readable single-line string."""
        if not isinstance(data, dict):
            return str(data)

        event_type = data.get("type")
        if event_type == "result" and (
            data.get("is_error") or str(data.get("subtype", "")).startswith("error")
        ):
            failed_result.set()
        if event_type == "assistant":
            content = data.get("message", {}).get("content", [])
            texts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    texts.append(part.get("text", ""))
            text = " ".join(t.strip() for t in texts if t and t.strip())
            return f"assistant: {text}" if text else "assistant (no text)"

        if event_type == "thinking":
            text = " ".join((data.get("text") or "").split())
            subtype = data.get("subtype")
            # Skip empty deltas to avoid noisy blank lines
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
                return "user (no text)"
            text = " ".join(text.split())
            return f"user: {text[:160]}{'...' if len(text) > 160 else ''}"

        if event_type == "system":
            model = data.get("model")
            cwd = data.get("cwd")
            return f"system init model={model} cwd={cwd}"

        # Fallback: compact json
        import json
        return json.dumps(data, ensure_ascii=False, separators=(",", ":"))

    def read_stream(stream, output_list, prefix, log_func):
        """Read from stream in a separate thread to avoid blocking"""
        import json
        import ast
        try:
            for line in iter(stream.readline, ''):
                if not line:
                    break
                raw_line = line.rstrip()

                # Try to parse as JSON (stream-json format)
                try:
                    data = json.loads(raw_line)
                    formatted = format_agent_event(data)
                    if formatted:
                        output_list.append(formatted)
                        log_func(f"{prefix} {formatted}")
                    continue
                except json.JSONDecodeError:
                    try:
                        data = ast.literal_eval(raw_line)
                        formatted = format_agent_event(data)
                        if formatted:
                            output_list.append(formatted)
                            log_func(f"{prefix} {formatted}")
                        continue
                    except Exception:
                        pass

                if raw_line.strip():
                    output_list.append(raw_line)
                    log_func(f"{prefix} {raw_line}")
        finally:
            stream.close()

    # Create threads to read stdout and stderr concurrently
    # This allows user interaction to work while we capture output
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

    # Start reading threads
    stdout_thread.start()
    stderr_thread.start()

    # Wait for process to complete
    timed_out = False
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        logger.warning(f"Cursor agent timed out after {timeout_seconds}s; terminating process")
        _stop_process(process)
    except BaseException:
        _stop_process(process)
        raise

    # Wait for output threads to finish reading
    stdout_thread.join(timeout=1)
    stderr_thread.join(timeout=1)

    # Log stderr summary if present
    if stderr_lines:
        logger.warning("=" * 80)
        logger.warning(f"Agent STDERR captured {len(stderr_lines)} lines")
        logger.warning("=" * 80)

    logger.info("=" * 80)
    logger.info(f"Agent completed with exit code: {process.returncode}")
    logger.info("=" * 80)

    # Return combined output
    output = "\n".join(stdout_lines)
    if stderr_lines:
        output += "\n=== STDERR ===\n" + "\n".join(stderr_lines)

    if timed_out:
        raise TimeoutError(f"Cursor timed out after {timeout_seconds}s; see agent logs")
    if process.returncode != 0:
        raise RuntimeError(f"Cursor exited with code {process.returncode}; see agent logs")
    if failed_result.is_set():
        raise RuntimeError("Cursor reported a failed result; see agent logs")
    return output
