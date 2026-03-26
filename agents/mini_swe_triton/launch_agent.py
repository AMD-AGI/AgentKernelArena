# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""
Mini-SWE Triton agent: raw single-round optimization via mini CLI.

No preprocessing, no COMMANDMENT, no profiler, no orchestrator.
Just gives the agent the kernel code, harness, and lets it optimize
freely. This is the baseline to measure what GEAK's structured
pipeline (preprocessing, profiling, multi-round orchestration,
heterogeneous task generation) adds on top.

Pipeline:
  1. Read kernel.py and build a simple task prompt
  2. mini --task <prompt> --test-command <harness> --repo <workspace>
     --num-parallel N --gpu-ids <gpus> --yolo --exit-immediately
"""
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
    timeout: int = 7200,
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
    Launch mini-SWE Triton agent: raw single-round parallel optimization.
    No preprocessing, no COMMANDMENT — just kernel code + harness + go.
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

    # Logs dir as sibling
    logs_dir = workspace_path.parent / f"{workspace_path.name}_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Build environment
    run_env = os.environ.copy()
    for k, v in (agent_config.get("geak_env") or {}).items():
        run_env[k] = str(v)

    gpu_ids = os.environ.get("GEAK_GPU_IDS", eval_config.get("gpu_ids", "0,1,2,3"))
    num_parallel = agent_config.get("agent", {}).get("num_parallel", 2)
    model = agent_config.get("agent", {}).get("model", "claude-opus-4-6")
    step_limit = agent_config.get("agent", {}).get("step_limit", 100)

    # PYTHONPATH for mini-swe-agent modules
    geak_src = os.environ.get("GEAK_SRC")
    if geak_src and Path(geak_src).is_dir():
        run_env["PYTHONPATH"] = f"{geak_src}:{run_env.get('PYTHONPATH', '')}"
    else:
        run_env["PYTHONPATH"] = f"/workspace/src:{run_env.get('PYTHONPATH', '')}"

    timeout = int(agent_config.get("timeout_seconds", 7200))

    logger.info("=" * 60)
    logger.info("  Mini-SWE Triton Agent (raw, no preprocessing)")
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

    # ── Build task prompt from kernel code directly ──────────────
    kernel_code = kernel_path.read_text()
    # Truncate if very large (keep first 3000 chars + last 1000)
    if len(kernel_code) > 4000:
        kernel_snippet = kernel_code[:3000] + "\n...\n" + kernel_code[-1000:]
    else:
        kernel_snippet = kernel_code

    task_prompt = f"""Optimize this Triton GPU kernel for maximum performance on AMD MI300X (gfx942/gfx950).

The kernel is at: {kernel_path.name}
The test harness is at: {harness_path.name}

To test your changes:
  python3 {harness_path.name} --correctness   # must pass
  python3 {harness_path.name} --benchmark     # measures performance

Rules:
- Only modify {kernel_path.name}
- Do NOT modify the test harness
- Correctness must pass after your changes
- Focus on real kernel-body optimizations (block sizes, memory access patterns,
  vectorization, loop unrolling, warp-level primitives)
- Target: AMD MI300X with gfx942/gfx950 architecture, 304 CUs, HBM3

Current kernel code:
```python
{kernel_snippet}
```
"""

    task_file = logs_dir / "_mini_task.md"
    task_file.write_text(task_prompt)

    # Build test command (correctness + benchmark)
    benchmark_iters = run_env.get("GEAK_BENCHMARK_ITERATIONS", "30")
    test_command = (
        f"python3 {harness_path} --correctness && "
        f"python3 {harness_path} --full-benchmark --iterations {benchmark_iters}"
    )

    # ── Initialize workspace as git repo ─────────────────────────
    git_env = {
        **run_env,
        "GIT_AUTHOR_NAME": "mini-swe",
        "GIT_AUTHOR_EMAIL": "mini-swe@amd.com",
        "GIT_COMMITTER_NAME": "mini-swe",
        "GIT_COMMITTER_EMAIL": "mini-swe@amd.com",
    }
    subprocess.run(["git", "init"], cwd=str(workspace_path),
                   capture_output=True, text=True)
    subprocess.run(["git", "add", "."], cwd=str(workspace_path),
                   capture_output=True, text=True)
    subprocess.run(["git", "commit", "-m", "baseline", "--allow-empty"],
                   cwd=str(workspace_path), capture_output=True, text=True,
                   env=git_env)

    # ── Run mini agent ───────────────────────────────────────────
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

    # ── Find best patch and apply to workspace ───────────────────
    best_applied = False

    # Check for patches in the output directory
    for patch_file in sorted(logs_dir.rglob("*.patch"), reverse=True):
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

    # Fallback: check if kernel.py was modified in any worktree
    if not best_applied:
        original_kernel = kernel_path.read_text()
        for wt_kernel in sorted(workspace_path.parent.rglob("kernel.py")):
            if wt_kernel == kernel_path:
                continue
            try:
                modified = wt_kernel.read_text()
                if modified != original_kernel:
                    kernel_path.write_text(modified)
                    logger.info(f"Copied modified kernel from {wt_kernel.parent.name}")
                    best_applied = True
                    break
            except OSError:
                continue

    if not best_applied:
        logger.warning("No applicable patch found from mini-swe output")

    return "\n".join(all_output)
