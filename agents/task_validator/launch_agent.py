# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import subprocess
import shutil
import logging
import threading
import shlex
from pathlib import Path
from typing import Any
import yaml
from agents import register_agent
from agents.task_validator.validation_prompt import build_validation_prompt


def _launch_claude_code(prompt: str, workspace: str, timeout_seconds: int, logger: logging.Logger) -> str:
    """Launch Claude Code CLI with the validation prompt."""
    AGENT = "claude"
    OPTIONS = (
        "--print "
        "--verbose "
        "--output-format stream-json "
        "--include-partial-messages "
        "--permission-mode bypassPermissions "
        "--dangerously-skip-permissions"
    )

    if not shutil.which(AGENT):
        raise RuntimeError(
            f"Command '{AGENT}' not found. Please ensure Claude Code CLI is installed and in your PATH."
        )

    quoted_prompt = shlex.quote(prompt)
    cmd = f"IS_SANDBOX=1 {AGENT} {OPTIONS} {quoted_prompt}"

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
                        if event_type == "stream_event":
                            ev = data.get("event", {})
                            ev_type = ev.get("type", "")
                            if ev_type in ("content_block_delta",):
                                # Skip noisy partial deltas in log
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
        process.wait(timeout=timeout_seconds)
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
    python_path = agent_config.get("python_path")

    # Inject python_path into eval_config for the prompt builder
    if python_path:
        eval_config.setdefault("agent", {})["python_path"] = python_path

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
    logger.info(f"Task config: {task_config_dir}")
    logger.info(f"Workspace: {workspace}")

    # Build validation prompt (custom, not using shared prompt_builder)
    prompt = build_validation_prompt(task_config_dir, workspace, eval_config)
    logger.info(f"Validation prompt built, length: {len(prompt)} characters")

    # Launch the chosen backend
    if backend == "claude_code":
        output = _launch_claude_code(prompt, workspace, timeout_seconds, logger)
    elif backend == "codex":
        # Placeholder for Codex backend
        raise NotImplementedError("Codex backend not yet implemented for task_validator")
    elif backend == "cursor":
        # Placeholder for Cursor backend
        raise NotImplementedError("Cursor backend not yet implemented for task_validator")
    else:
        raise ValueError(f"Unknown backend: {backend}. Supported: claude_code, codex, cursor")

    # Verify that validation_report.yaml was generated
    report_path = Path(workspace) / "validation_report.yaml"
    if report_path.exists():
        logger.info(f"Validation report generated: {report_path}")
    else:
        logger.warning(f"WARNING: validation_report.yaml was NOT generated in {workspace}")

    return output
