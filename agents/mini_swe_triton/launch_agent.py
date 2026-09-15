# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Legacy mini CLI compatibility; schema-v2 tasks are explicitly unsupported.

The required GEAK mini fork is separate from the current GEAK Workflow engine.
See README.md for the inspected CLI and the missing v2 integration contract.
"""
import logging
import os
import shlex
import signal
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any

import yaml

from agents import register_agent
from src.task_spec import resolve_task_path


class MiniSweCapabilityError(RuntimeError):
    """The selected task/runtime has no implemented mini adapter."""


def _require_legacy_task(task_config: dict[str, Any]) -> None:
    # Check before filenames, dependencies, git initialization, or output writes.
    # A framework context also prevents a missing version from falling through.
    if ("schema_version" in task_config or "candidate" in task_config
            or "evaluation" in task_config or "ARENA_TASK_CONTEXT" in os.environ):
        raise MiniSweCapabilityError(
            "MINI_SWE_V2_UNSUPPORTED: mini_swe_triton has no verified adapter for "
            "TaskSpec/ARENA_TASK_CONTEXT and task-owned v2 actions. All retained "
            "Arena tasks use v2. This launcher only retains external legacy "
            "single-file task compatibility; it will not infer kernel.py or "
            "substitute the GEAK Workflow engine. See agents/mini_swe_triton/README.md."
        )


def _legacy_runtime_source() -> Path:
    value = os.environ.get("GEAK_SRC")
    source = Path(value) if value else None
    if (source is None or not source.is_absolute()
            or not (source / "minisweagent" / "run" / "mini.py").is_file()):
        raise MiniSweCapabilityError(
            "MINI_SWE_RUNTIME_UNAVAILABLE: GEAK_SRC must be the absolute source "
            "directory of the legacy GEAK mini fork containing "
            "minisweagent/run/mini.py. A current GEAK Workflow checkout or an "
            "arbitrary directory is not that runtime."
        )
    return source.resolve()


def _legacy_paths(task_config: dict[str, Any], workspace: Path) -> tuple[Path, Path]:
    sources = task_config.get("source_file_path", ["kernel.py"])
    if not isinstance(sources, list) or len(sources) != 1:
        raise MiniSweCapabilityError(
            "Legacy mini requires exactly one source_file_path; multifile tasks "
            "need a verified v2 adapter."
        )
    kernel = resolve_task_path(workspace, sources[0], must_exist=True)
    harness = resolve_task_path(
        workspace, task_config.get("harness_path", "test_kernel_harness.py"), must_exist=True,
    )
    if not kernel.is_file() or not harness.is_file() or kernel == harness:
        raise ValueError("Legacy mini requires separate regular kernel and harness files")
    return kernel, harness


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
    cmd: list[str],
    *,
    env: dict[str, str],
    cwd: str,
    label: str,
    logger: logging.Logger,
    timeout: float = 7200,
) -> tuple[int, list[str], list[str]]:
    logger.info("[%s] Running argv: %r", label, cmd)
    logger.info(f"[{label}] cwd: {cwd}")

    proc = subprocess.Popen(
        cmd, shell=False, start_new_session=True,
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

    timed_out = False
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.warning(f"[{label}] Timed out after {timeout}s; killing")
        timed_out = True
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()

    t_out.join(timeout=5)
    t_err.join(timeout=5)

    logger.info(f"[{label}] exit code: {proc.returncode}")
    if timed_out:
        raise subprocess.TimeoutExpired(cmd, timeout, output="\n".join(stdout_lines),
                                        stderr="\n".join(stderr_lines))
    return proc.returncode, stdout_lines, stderr_lines


@register_agent("mini_swe_triton")
def launch_agent(eval_config: dict[str, Any], task_config_dir: str, workspace: str) -> str:
    """Run the legacy CLI, or reject unsupported task contracts before side effects."""
    logger = logging.getLogger(__name__)

    with open(task_config_dir) as f:
        task_config = yaml.safe_load(f)
    if not isinstance(task_config, dict):
        raise ValueError("mini_swe_triton task config must be a mapping")
    _require_legacy_task(task_config)
    geak_src = _legacy_runtime_source()

    config_path = Path(__file__).with_name("agent_config.yaml")
    with config_path.open() as f:
        agent_config = yaml.safe_load(f) or {}

    workspace_path = Path(workspace).resolve()
    kernel_path, harness_path = _legacy_paths(task_config, workspace_path)
    kernel_relative = kernel_path.relative_to(workspace_path).as_posix()
    harness_relative = harness_path.relative_to(workspace_path).as_posix()

    timeout = agent_config.get("timeout_seconds", 7200)
    if type(timeout) is not int or timeout <= 0:
        raise ValueError("mini timeout_seconds must be a positive integer")

    # Each invocation owns a fresh output directory; prior artifacts stay intact.
    logs_dir = Path(tempfile.mkdtemp(prefix=f"{workspace_path.name}_mini_",
                                    dir=workspace_path.parent))

    # Build environment
    run_env = os.environ.copy()
    for k, v in (agent_config.get("geak_env") or {}).items():
        run_env[k] = str(v)

    gpu_ids = os.environ.get("GEAK_GPU_IDS", eval_config.get("gpu_ids", "0,1,2,3"))
    num_parallel = agent_config.get("agent", {}).get("num_parallel", 2)
    model = agent_config.get("agent", {}).get("model", "claude-opus-4-6")
    step_limit = agent_config.get("agent", {}).get("step_limit", 100)

    run_env["PYTHONPATH"] = f"{geak_src}:{run_env.get('PYTHONPATH', '')}"

    logger.info("=" * 60)
    logger.info("  Mini-SWE Triton Agent (legacy compatibility)")
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

    # ── Build task prompt from kernel code directly ──────────────
    kernel_code = kernel_path.read_text()
    # Truncate if very large (keep first 3000 chars + last 1000)
    if len(kernel_code) > 4000:
        kernel_snippet = kernel_code[:3000] + "\n...\n" + kernel_code[-1000:]
    else:
        kernel_snippet = kernel_code

    task_prompt = f"""Optimize this Triton GPU kernel for the configured GPU: {eval_config.get('target_gpu_model', 'inspect the runtime GPU')}.

The kernel is at: {kernel_relative}
The test harness is at: {harness_relative}

To test your changes:
  python3 {shlex.quote(harness_relative)} --correctness   # must pass
  python3 {shlex.quote(harness_relative)} --benchmark     # measures performance

Rules:
- Only modify {kernel_relative}
- Do NOT modify the test harness
- Correctness must pass after your changes
- Focus on real kernel-body optimizations (block sizes, memory access patterns,
  vectorization, loop unrolling, warp-level primitives)
- Preserve the workload, numerical checks and timing policy.

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
        f"python3 {shlex.quote(harness_relative)} --correctness && "
        f"python3 {shlex.quote(harness_relative)} --full-benchmark "
        f"--iterations {shlex.quote(benchmark_iters)}"
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
                   capture_output=True, text=True, check=True, timeout=60, env=git_env)
    subprocess.run(["git", "add", "."], cwd=str(workspace_path),
                   capture_output=True, text=True, check=True, timeout=60, env=git_env)
    subprocess.run(["git", "commit", "-m", "baseline", "--allow-empty"],
                   cwd=str(workspace_path), capture_output=True, text=True,
                   env=git_env, check=True, timeout=60)

    # ── Run mini agent ───────────────────────────────────────────
    mini_cmd = [
        "python3", "-m", "minisweagent.run.mini", "--task", str(task_file),
        "--test-command", test_command, "--repo", str(workspace_path),
        "--num-parallel", str(num_parallel), "--gpu-ids", str(gpu_ids),
        "--model", str(model), "--yolo", "--exit-immediately", "-o", str(logs_dir),
        "--cost-limit", "0",
    ]
    try:
        rc_mini, out_mini, err_mini = _run_step(
            mini_cmd, env=run_env, cwd=str(workspace_path),
            label="mini-swe", logger=logger, timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        (logs_dir / "stdout.log").write_text(exc.stdout or "")
        (logs_dir / "stderr.log").write_text(exc.stderr or "")
        raise RuntimeError(f"mini-swe timed out; invocation logs: {logs_dir}") from exc
    (logs_dir / "stdout.log").write_text("\n".join(out_mini))
    (logs_dir / "stderr.log").write_text("\n".join(err_mini))
    if rc_mini != 0:
        raise RuntimeError(f"mini-swe exited with code {rc_mini}; invocation logs: {logs_dir}")

    # The inspected legacy CLI applies its selected result to --repo itself.
    # Never guess a winning patch or import another invocation's candidate.
    logger.info("mini-swe completed; Arena evaluates only the retained workspace files")
    return "\n".join(out_mini)
