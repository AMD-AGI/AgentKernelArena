# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import subprocess
import shutil
import logging
import threading
import os
import shlex
import re
from pathlib import Path
from datetime import datetime
from typing import Any
import json
import socket
import yaml
from agents import register_agent
from src.preprocessing import setup_repo_from_config
from agents.geak_v3.geak_pre_process import (
    simple_prompt_builder,
    integrate_agent_config,
    copy_python_bindings,
)

def _append_jsonl_record(path: Path, record: dict[str, Any], logger: logging.Logger) -> None:
    """
    Append one JSON object per line (JSONL).

    Uses a file lock (fcntl) on Linux to avoid interleaved writes across processes.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        try:
            import fcntl  # type: ignore
        except Exception:
            fcntl = None  # type: ignore

        with open(path, "a", encoding="utf-8") as f:
            if fcntl is not None:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                except Exception:
                    pass
            f.write(line + "\n")
            if fcntl is not None:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass
    except Exception as e:
        logger.warning(f"Failed to write agent invocation record to {path}: {e}")


def _get_invocation_log_path() -> Path:
    """
    Unified file path to store invocation records.

    Priority:
      1) AKA_AGENT_CMD_LOG env var (per-run path)
      2) <project_root>/logs/agent_invocations.jsonl
    """
    env_path = os.environ.get("AKA_AGENT_CMD_LOG")
    if env_path:
        return Path(env_path).expanduser().resolve()

    project_root = Path(__file__).resolve().parent.parent.parent
    return (project_root / "logs" / "agent_invocations.jsonl").resolve()


def write_debug_script(workspace: str, cmd: str, agent: str) -> None:
    """Optionally write the invocation command to a shell script for debugging."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    script_file = Path(workspace) / f"run_agent_{timestamp}.sh"

    script_lines = [
        "#!/bin/bash",
        f"# Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"# Workspace: {workspace}",
        f"# Agent: {agent}",
        "",
        f"cd {workspace}",
        cmd,
    ]

    script_file.write_text("\n".join(script_lines) + "\n")
    os.chmod(script_file, 0o755)


@register_agent("geak_v3")
def launch_agent(eval_config: dict[str, Any], task_config_dir: str, workspace: str) -> str:
    """
    Launch geak_v3 agent using mini-SWE-agent with real-time output streaming.

    Args:
        eval_config: Evaluator settings passed from main (includes task metadata like task_type)
        task_config_dir: Path to the task configuration used to build the prompt
        workspace: Workspace directory where the agent will run and read/write files

    Returns:
        str: Combined agent output (stdout plus stderr summary if present)
    """
    # Load agent config (support override via env var)
    config_path_env = os.environ.get("GEAK_AGENT_CONFIG")
    if config_path_env:
        config_path = Path(config_path_env)
    else:
        config_path = Path(__file__).with_name("agent_config.yaml")
    with config_path.open("r") as f:
        agent_config = yaml.safe_load(f) or {}
    logger = logging.getLogger(__name__)

    # Get run configuration
    run_config = agent_config.get("run", {})
    
    AGENT = run_config.get("cmd", "geak")
    
    # Get configs string (e.g., '-c geak.yaml --yolo --num-parallel=2 --gpu-ids=0,1')
    OPTIONS = run_config.get("configs", "")
    
    # Replace relative config file path with absolute path (e.g., '-c geak.yaml' -> '-c /abs/path/geak.yaml')
    agent_dir = Path(__file__).parent
    def replace_config_path(match):
        config_file = match.group(1)
        abs_path = agent_dir / config_file
        return f"-c {abs_path!s}"
    OPTIONS = re.sub(r'-c\s+(\S+)', replace_config_path, OPTIONS)

    # Check if the command exists
    if not shutil.which(AGENT):
        raise RuntimeError(
            f"Command '{AGENT}' not found. Please ensure it is installed and in your PATH."
        )

    # Load task configuration
    task_config_path = Path(task_config_dir)
    with open(task_config_path, 'r') as f:
        task_config = yaml.safe_load(f)

    # Convert the workspace path to an absolute path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    workspace = os.path.abspath(os.path.join(project_root, workspace))

    # Setup repo from config if repo_url is present
    repo_path = setup_repo_from_config(task_config_dir, Path(workspace), logger)
    if repo_path:
        logger.info(f"Repository cloned to: {repo_path}")
        # Note: We use workspace (not repo_path) as --repo because test_command
        # (e.g., 'python3 scripts/task_runner.py') is relative to workspace root,
        # not the cloned repo subdirectory.
        OPTIONS += f" --repo={shlex.quote(workspace)}"

    # Copy python_bindings to workspace
    copy_python_bindings(task_config_dir, workspace, logger)

    # Build simplified prompt (only instructions + workspace info)
    prompt = simple_prompt_builder(task_config_dir, workspace, logger)
    prompt = integrate_agent_config(prompt, agent_config)

    # Write prompt to a temporary file (mini agent reads from file if path exists)
    prompt_file = Path(workspace) / "task_prompt.md"
    prompt_file.write_text(prompt, encoding="utf-8")
    logger.info(f"Wrote task prompt to: {prompt_file}")

    # Put optimization_logs outside workspace to avoid recursive copying when creating worktrees
    # Use a sibling directory: workspace_dir_logs/
    workspace_path = Path(workspace)
    logs_dir = workspace_path.parent / f"{workspace_path.name}_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    cmd = f"{AGENT} {OPTIONS} -t {shlex.quote(str(prompt_file))} -o {shlex.quote(str(logs_dir))}"

    # Persist the exact invocation for debugging (unified JSONL file)
    _append_jsonl_record(
        _get_invocation_log_path(),
        {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "cwd": os.getcwd(),
            "task_name": os.environ.get("AKA_TASK_NAME"),
            "agent_launcher": "agents/geak_v3/launch_agent.py",
            "run_cmd": AGENT,
            "run_configs": run_config.get("configs", ""),
            "options_final": OPTIONS,
            "cmd_final": cmd,
            "agent_config_path": str(config_path.resolve()),
            "task_config_dir": task_config_dir,
            "workspace": workspace,
            "prompt_file": str(prompt_file),
            "patch_output_dir": str(logs_dir),
            "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
            "rocr_visible_devices": os.environ.get("ROCR_VISIBLE_DEVICES"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        logger,
    )

    # Enable to save the command to a shell script for manual replay/debugging.
    if False:
        write_debug_script(workspace, cmd, AGENT)
        logger.info("Debug script written; skipping live run.")
        return ""

    logger.info(f"Running command: {cmd}")
    logger.info("=" * 80)
    logger.info("Agent Output (streaming):")
    logger.info("=" * 80)

    # Give the agent a hard stop to avoid blocking downstream tasks
    timeout_seconds = int(agent_config.get("timeout_seconds", 3600))

    # Use Popen for real-time output streaming
    process = subprocess.Popen(
        cmd,
        shell=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=workspace,
        bufsize=1
    )

    # Close stdin immediately
    if process.stdin:
        process.stdin.close()

    # Collect output while streaming
    stdout_lines = []
    stderr_lines = []

    def format_agent_event(data):
        """Convert cursor stream-json payloads into a readable single-line string."""
        if not isinstance(data, dict):
            return str(data)

        event_type = data.get("type")
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
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        logger.warning(f"Agent timed out after {timeout_seconds}s; terminating process")
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("Force killing agent process")
            process.kill()

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

    # Apply best patch to original workspace so evaluator sees optimized code
    _apply_best_patch_to_workspace(workspace, logs_dir, logger)

    # Return combined output
    output = "\n".join(stdout_lines)
    if stderr_lines:
        output += "\n=== STDERR ===\n" + "\n".join(stderr_lines)

    return output


def _apply_best_patch_to_workspace(workspace: str, logs_dir: Path, logger: logging.Logger) -> bool:
    """
    Apply the best patch from logs_dir to the original workspace.
    
    This ensures the centralized evaluator (in main.py) evaluates the optimized code,
    not the original baseline code.
    
    Args:
        workspace: Original workspace directory
        logs_dir: Logs directory containing best_results.json and patch files
        logger: Logger instance
        
    Returns:
        True if patch was applied successfully, False otherwise
    """
    import json
    
    # Find best_results.json
    best_results_file = logs_dir / "best_results.json"
    if not best_results_file.exists():
        logger.warning("No best_results.json found, skipping patch application")
        return False
    
    try:
        with open(best_results_file, 'r') as f:
            best_results = json.load(f)
        
        patch_file = best_results.get('best_patch_file')
        if not patch_file or not Path(patch_file).exists():
            logger.warning(f"Best patch file not found: {patch_file}")
            return False
        
        logger.info("=" * 80)
        logger.info(f"Applying best patch to workspace: {patch_file}")
        logger.info("=" * 80)
        
        # Try git apply first (works if workspace is a git repo)
        result = subprocess.run(
            ["git", "apply", "--check", str(patch_file)],
            cwd=workspace,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            # Patch can be applied cleanly with git
            result = subprocess.run(
                ["git", "apply", str(patch_file)],
                cwd=workspace,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                logger.info(f"Successfully applied patch with git apply")
                return True
            else:
                logger.warning(f"git apply failed: {result.stderr}")
        
        # Fallback to patch command
        result = subprocess.run(
            ["patch", "-p1", "--dry-run", "-i", str(patch_file)],
            cwd=workspace,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            result = subprocess.run(
                ["patch", "-p1", "-i", str(patch_file)],
                cwd=workspace,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                logger.info(f"Successfully applied patch with patch command")
                return True
            else:
                logger.warning(f"patch command failed: {result.stderr}")
        else:
            logger.warning(f"Patch dry-run failed: {result.stderr}")
        
        return False
        
    except Exception as e:
        logger.error(f"Error applying best patch: {e}")
        return False
