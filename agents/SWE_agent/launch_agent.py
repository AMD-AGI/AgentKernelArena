# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import subprocess
import shutil
import logging
import threading
import os
import shlex
from pathlib import Path
from datetime import datetime
from typing import Any
import yaml
from agents import register_agent
from src.module_registration import AgentType, load_prompt_builder


def integrate_agent_config(prompt, agent_config: dict[str, Any]) -> str:
    """
    Integrate agent config into prompt.
    """
    max_iters = agent_config.get("max_iterations")
    if max_iters is not None:
        prompt = prompt.rstrip() + f"\n\nFor this optimization, you must iterate up to {max_iters} versions."
    python_path = agent_config.get("python_path")
    if python_path:
        prompt = prompt.rstrip() + f"\n\nUse this Python interpreter: `{python_path}`."
    return prompt

def write_debug_script(workspace: str, cmd: str, agent: str) -> None:
    """Optionally write the invocation command to a shell script for debugging."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    script_file = f"run_agent_{timestamp}.sh"

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


@register_agent("swe_agent")
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
    AGENT = "mini"
    # Use stream-json format with partial output for real-time streaming

    config_path = Path(__file__).with_name("agent_config.yaml")
    with config_path.open("r") as f:
        agent_config = yaml.safe_load(f) or {}
    logger = logging.getLogger(__name__)

    # Load task configuration
    task_config_path = Path(task_config_dir)
    with open(task_config_path, 'r') as f:
        task_config = yaml.safe_load(f)


    if task_config["task_type"] == "instruction2triton":
        mini_yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mini_tritonbench.yaml")
    else:
        mini_yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mini.yaml")
    OPTIONS = f"-c {mini_yaml_path!s}"

    # Check if the command exists
    if not shutil.which(AGENT):
       raise RuntimeError(
           f"Command '{AGENT}' not found. Please ensure cursor-agent is installed and in your PATH."
       )
    
    prompt_builder = load_prompt_builder(AgentType.SWE_AGENT, logger)

    # Convert the workspace path to an absolute path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    workspace = os.path.abspath(os.path.join(project_root, workspace))

    # iterate over the tasks to check if triton/tritonbench is in the tasks
    if any("triton/tritonbench" in task for task in eval_config["tasks"]):
        tritonbench_script_path = Path(task_config_dir).parent / "python_bindings" / "tritonbench.py"
        # make a dir for the target path
        os.makedirs(os.path.join(workspace, "python_bindings"), exist_ok=True)
        # copy the script python_bindings/tritonbench.py into the workspace
        shutil.copy(tritonbench_script_path, os.path.join(workspace, "python_bindings", "tritonbench.py"))
    if any("rocprim" in task for task in eval_config["tasks"]):
        for task in eval_config["tasks"]:
            if "rocprim" not in task:
                continue
            repo_dir = Path(workspace) / "tasks" / task / "rocPRIM"
            if (repo_dir / ".git").exists():
                logger.info(f"Repository already exists at {repo_dir}, skipping clone")
                continue
            if repo_dir.exists():
                logger.info(f"Repository directory already exists at {repo_dir}, skipping clone")
                continue
            repo_dir.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "clone", "https://github.com/ROCm/rocPRIM.git", str(repo_dir)],
                check=True,
            )
        test_correctness_benchmark_path = Path(task_config_dir).parent / "python_bindings" / "test_correctness_benchmark.py"
        # make a dir for the target path
        os.makedirs(os.path.join(workspace, "python_bindings"), exist_ok=True)
        # copy the script python_bindings/test_correctness_benchmark.py into the workspace
        shutil.copy(test_correctness_benchmark_path, os.path.join(workspace, "python_bindings", "test_correctness_benchmark.py"))
        
    prompt = prompt_builder(task_config_dir, workspace, eval_config, logger)

    prompt = integrate_agent_config(prompt, agent_config)
    quoted_prompt = shlex.quote(prompt)
    cmd = f"{AGENT} {OPTIONS} -t {quoted_prompt} --yolo --output {workspace}/output.traj.json"

    # Enable to save the command to a shell script for manual replay/debugging.
    if False:
        write_debug_script(workspace, cmd, AGENT)
        logger.info("Debug script written; skipping live run.")
        return ""
    
    logger.info(f"Running command: {cmd}")
    logger.info("=" * 80)
    logger.info("Agent Output (streaming):")
    logger.info("=" * 80)

    # Give the agent a hard stop to avoid blocking downstream tasks if it
    # keeps waiting for interactive input after finishing its work.
    timeout_seconds = int(agent_config.get("timeout_seconds", 3600))

    # Use Popen for real-time output streaming with interactive input support
    process = subprocess.Popen(
        cmd,
        shell=True,  # nosec B602 -- shell=True is required to launch agent process
        stdin=subprocess.PIPE,  # Keep stdin closed so the agent exits when done
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=workspace,
        bufsize=1  # Line buffered
    )

    # Close stdin immediately; leaving it attached keeps the agent alive waiting
    # for more user messages even after it reports completion.
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
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        logger.warning(f"Cursor agent timed out after {timeout_seconds}s; terminating process")
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("Force killing cursor agent process")
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

    # Return combined output
    output = "\n".join(stdout_lines)
    if stderr_lines:
        output += "\n=== STDERR ===\n" + "\n".join(stderr_lines)

    return output
