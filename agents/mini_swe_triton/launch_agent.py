# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""
Mini-SWE Triton agent: single-round parallel optimization via mini CLI.

Apple-to-apple comparison with GEAK-v3:
- Same preprocessing (geak-preprocess generates COMMANDMENT, baseline, profile)
- Same harness (test_kernel_harness.py --correctness / --full-benchmark)
- Same number of parallel agents (4)
- Same model (claude-opus-4-6)
- Difference: single round (mini) vs multi-round orchestration (GEAK)
  This isolates the value of GEAK's multi-round iteration and heterogeneous
  task generation.

Pipeline:
  1. geak-preprocess <kernel> --harness <harness> -o <logs_dir>
  2. mini --task <COMMANDMENT> --test-command <harness> --repo <workspace>
     --num-parallel N --gpu-ids <gpus> --yolo --exit-immediately -o <logs_dir>
"""
import json
import logging
import os
import subprocess
import threading
from pathlib import Path
from typing import Any

import yaml

from agents import register_agent


def _read_stream(stream, lines: list, prefix: str, log_func):
    try:
        for line in iter(stream.readline, ""):
            if not line:
                break
            raw = line.rstrip()
            if raw.strip():
                lines.append(raw)
                log_func(f"{prefix} {raw}")
    finally:
        stream.close()


def _run_step(
    cmd: str,
    *,
    env: dict[str, str],
    cwd: str,
    label: str,
    logger: logging.Logger,
    timeout: int = 14400,
) -> tuple[int, list[str], list[str]]:
    logger.info(f"[{label}] Running: {cmd}")
    logger.info(f"[{label}] cwd: {cwd}")

    proc = subprocess.Popen(
        cmd, shell=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, cwd=cwd, env=env, bufsize=1,
    )

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    t_out = threading.Thread(
        target=_read_stream,
        args=(proc.stdout, stdout_lines, f"[{label}]", logger.info),
        daemon=True,
    )
    t_err = threading.Thread(
        target=_read_stream,
        args=(proc.stderr, stderr_lines, f"[{label} ERR]", logger.warning),
        daemon=True,
    )
    t_out.start()
    t_err.start()

    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.warning(f"[{label}] Timed out after {timeout}s; killing")
        proc.kill()

    t_out.join(timeout=5)
    t_err.join(timeout=5)

    logger.info(f"[{label}] exit code: {proc.returncode}")
    return proc.returncode, stdout_lines, stderr_lines


@register_agent("mini_swe_triton")
def launch_agent(eval_config: dict[str, Any], task_config_dir: str, workspace: str) -> str:
    """
    Launch mini-SWE Triton agent: preprocess then single-round parallel optimization.
    """
    logger = logging.getLogger(__name__)

    config_path = Path(__file__).with_name("agent_config.yaml")
    with config_path.open() as f:
        agent_config = yaml.safe_load(f) or {}

    with open(task_config_dir) as f:
        task_config = yaml.safe_load(f) or {}

    workspace_path = Path(workspace).resolve()
    kernel_path = workspace_path / (task_config.get("source_file_path", ["kernel.py"])[0])
    harness_path = workspace_path / task_config.get("harness_path", "test_kernel_harness.py")

    if not kernel_path.is_file():
        raise FileNotFoundError(f"Kernel not found: {kernel_path}")
    if not harness_path.is_file():
        raise FileNotFoundError(f"Harness not found: {harness_path}")

    # Logs dir as sibling (same pattern as geak_v3_triton)
    logs_dir = workspace_path.parent / f"{workspace_path.name}_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Build environment
    run_env = os.environ.copy()
    for k, v in (agent_config.get("geak_env") or {}).items():
        run_env[k] = str(v)

    gpu_ids = os.environ.get("GEAK_GPU_IDS", eval_config.get("gpu_ids", "0,1,2,3"))
    first_gpu = gpu_ids.split(",")[0]
    num_parallel = agent_config.get("agent", {}).get("num_parallel", 4)
    model = agent_config.get("agent", {}).get("model", "claude-opus-4-6")
    step_limit = agent_config.get("agent", {}).get("step_limit", 200)

    # PYTHONPATH for GEAK modules
    geak_src = os.environ.get("GEAK_SRC")
    if geak_src and Path(geak_src).is_dir():
        run_env["PYTHONPATH"] = f"{geak_src}:{run_env.get('PYTHONPATH', '')}"
    else:
        run_env["PYTHONPATH"] = f"/workspace/src:{run_env.get('PYTHONPATH', '')}"

    timeout = int(agent_config.get("timeout_seconds", 14400))

    logger.info("=" * 60)
    logger.info("  Mini-SWE Triton Agent (single-round parallel)")
    logger.info("=" * 60)
    logger.info(f"  kernel:       {kernel_path}")
    logger.info(f"  harness:      {harness_path}")
    logger.info(f"  workspace:    {workspace_path}")
    logger.info(f"  logs_dir:     {logs_dir}")
    logger.info(f"  gpu_ids:      {gpu_ids}")
    logger.info(f"  num_parallel: {num_parallel}")
    logger.info(f"  model:        {model}")
    logger.info(f"  step_limit:   {step_limit}")
    logger.info("=" * 60)

    all_output: list[str] = []

    # ── Step 1: geak-preprocess (same as GEAK-v3) ───────────────
    # This generates COMMANDMENT.md, baseline metrics, profiling data.
    # The mini agent uses COMMANDMENT as its task prompt.
    preprocess_cmd = (
        f"python3 -m minisweagent.run.preprocess.preprocessor"
        f" {kernel_path}"
        f" --harness {harness_path}"
        f" --gpu {first_gpu}"
        f" --model {model}"
        f" -o {logs_dir}"
    )

    rc_pre, out_pre, err_pre = _run_step(
        preprocess_cmd, env=run_env, cwd=str(workspace_path),
        label="preprocess", logger=logger, timeout=3600,
    )
    all_output.extend(out_pre)

    if rc_pre != 0:
        logger.warning(f"geak-preprocess exited with code {rc_pre}")
        all_output.extend(err_pre)

    # ── Step 2: Build task prompt from COMMANDMENT ───────────────
    commandment_path = logs_dir / "COMMANDMENT.md"
    task_prompt = f"Optimize the Triton kernel at {kernel_path.name} for maximum performance on AMD MI300X GPU."

    if commandment_path.exists():
        commandment_text = commandment_path.read_text()
        task_prompt = commandment_text
        logger.info(f"Using COMMANDMENT.md as task ({len(commandment_text)} chars)")
    else:
        logger.warning("COMMANDMENT.md not found, using generic prompt")

    # Write task to a temp file (avoids shell quoting issues)
    task_file = logs_dir / "_mini_task.md"
    task_file.write_text(task_prompt)

    # Build test command from COMMANDMENT sections
    test_cmd_parts = []
    # Correctness
    test_cmd_parts.append(f"python3 {harness_path} --correctness")
    # Benchmark
    benchmark_iters = run_env.get("GEAK_BENCHMARK_ITERATIONS", "30")
    test_cmd_parts.append(
        f"python3 {harness_path} --full-benchmark --iterations {benchmark_iters}"
    )
    test_command = " && ".join(test_cmd_parts)

    # ── Step 3: Run mini agent ───────────────────────────────────
    # Initialize workspace as git repo for worktree management
    subprocess.run(["git", "init"], cwd=str(workspace_path),
                   capture_output=True, text=True)
    subprocess.run(["git", "add", "."], cwd=str(workspace_path),
                   capture_output=True, text=True)
    subprocess.run(
        ["git", "commit", "-m", "baseline", "--allow-empty"],
        cwd=str(workspace_path), capture_output=True, text=True,
        env={**run_env, "GIT_AUTHOR_NAME": "geak", "GIT_AUTHOR_EMAIL": "geak@amd.com",
             "GIT_COMMITTER_NAME": "geak", "GIT_COMMITTER_EMAIL": "geak@amd.com"},
    )

    mini_cmd = (
        f"python3 -m minisweagent.run.mini"
        f" --task {task_file}"
        f" --test-command '{test_command}'"
        f" --repo {workspace_path}"
        f" --num-parallel {num_parallel}"
        f" --gpu-ids {gpu_ids}"
        f" --model {model}"
        f" --yolo"
        f" --exit-immediately"
        f" -o {logs_dir}"
        f" --cost-limit 0"
    )

    rc_mini, out_mini, err_mini = _run_step(
        mini_cmd, env=run_env, cwd=str(workspace_path),
        label="mini-swe", logger=logger, timeout=timeout,
    )
    all_output.extend(out_mini)

    if rc_mini != 0:
        logger.warning(f"mini-swe exited with code {rc_mini}")
        all_output.extend(err_mini)

    # ── Step 4: Find best patch and apply to workspace ───────────
    # mini-swe writes patches to logs_dir/patches/ or similar
    # Look for best_results.json or patch files
    best_applied = False
    for patch_dir in sorted(logs_dir.glob("**/patches"), reverse=True):
        for patch_file in sorted(patch_dir.glob("*.patch"), reverse=True):
            try:
                result = subprocess.run(
                    ["git", "apply", "--check", str(patch_file)],
                    cwd=str(workspace_path), capture_output=True, text=True,
                )
                if result.returncode == 0:
                    subprocess.run(
                        ["git", "apply", str(patch_file)],
                        cwd=str(workspace_path), capture_output=True, text=True,
                    )
                    logger.info(f"Applied patch: {patch_file.name}")
                    best_applied = True
                    break
            except Exception as e:
                logger.warning(f"Patch {patch_file.name} failed: {e}")
        if best_applied:
            break

    if not best_applied:
        logger.warning("No applicable patch found from mini-swe output")

    return "\n".join(all_output)
