# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""AgentKernelArena adapter for GEAK v4's deterministic kernel workflow.

GEAK v4 optimizes a kernel by running its JavaScript ``kernel_workflow`` through
Claude Code's dynamic ``Workflow`` tool. On current Claude builds that Workflow
runs as a *background* task, so the SDK / background-task lifecycle is handled by
the sibling ``workflow_runner.py`` (a kernel-scoped analogue of GEAK's own
``interface/run_e2e.py``). This launcher stays deliberately thin and, like the
``forge`` / ``claude_code`` agents, relies purely on the environment:

  * gate unsupported task types,
  * resolve the GEAK checkout (``GEAK_V4_WORKFLOW_DIR``) and ``claude`` (PATH),
  * write a versioned handoff and run ``workflow_runner.py`` as a subprocess,
  * let GEAK apply its Director-validated patch straight into the workspace
    (``apply_to_original="true"``).

Workspace integrity is the Arena harness's job, not the launcher's: main.py
snapshots the harness before the agent runs, verifies it afterwards,
re-materializes the perf helpers, and independently re-scores the kernel. That is
why this adapter needs no disposable-copy / manifest / patch-reimport machinery.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from agents import register_agent
from src.module_registration import AgentType, load_prompt_builder


_JSON_SIZE_LIMIT = 8 * 1024 * 1024
_PROCESS_OUTPUT_LIMIT = 4 * 1024 * 1024


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    if len(content) > _JSON_SIZE_LIMIT:
        return None
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _declared_sources(task_config: dict[str, Any], workspace: Path) -> list[str]:
    """Return the workspace-relative source files used to steer the optimizer.

    Only guidance: GEAK edits the kernel and the Arena harness re-scores it, so
    this is not a security boundary. We fail early if the declared anchor source
    is absent (mirrors ``forge``'s kernel-file resolution) and otherwise pass the
    names through to the "optimize only these files" prompt note.
    """
    raw = task_config.get("source_file_path") or []
    values = [raw] if isinstance(raw, str) else list(raw)
    sources = [value.strip() for value in values if isinstance(value, str) and value.strip()]
    if sources and not (workspace / sources[0]).exists():
        raise FileNotFoundError(
            f"declared source_file_path not found in workspace: {sources[0]}"
        )
    return sources


def _logical_gpu_ids(eval_config: dict[str, Any]) -> str:
    """Return GPU IDs in the process-visible namespace.

    Arena parallel workers mask one physical GPU with ROCR_VISIBLE_DEVICES and
    expose it as logical HIP/CUDA device 0. GEAK's gpu_lock wrapper rewrites
    HIP_VISIBLE_DEVICES again, so forwarding the host ID would hide the GPU.
    """
    if os.environ.get("AGENT_KERNEL_ARENA_HOST_GPU_ID") is not None:
        return "0"

    visible = (
        os.environ.get("HIP_VISIBLE_DEVICES")
        or os.environ.get("CUDA_VISIBLE_DEVICES")
    )
    if visible:
        count = len([part for part in visible.split(",") if part.strip()])
        if count:
            return ",".join(str(index) for index in range(count))

    override = os.environ.get("GEAK_V4_GPU_IDS")
    configured = override if override is not None else eval_config.get("gpu_ids", "0")
    if isinstance(configured, (list, tuple)):
        configured = ",".join(str(item) for item in configured)
    return str(configured)


def _process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_group_exit(process: subprocess.Popen[str], pgid: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        process.poll()
        if not _process_group_exists(pgid):
            return True
        time.sleep(0.1)
    return not _process_group_exists(pgid)


def _terminate_process_group(
    process: subprocess.Popen[str],
    logger: logging.Logger,
) -> None:
    pgid = process.pid
    if not _process_group_exists(pgid):
        process.poll()
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    if _wait_group_exit(process, pgid, 10):
        return
    logger.warning("Force killing GEAK runner process group")
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return
    _wait_group_exit(process, pgid, 5)


def _stream_pipe(stream, prefix: str, output: list[str], log) -> None:
    captured = 0
    truncated = False
    try:
        while True:
            chunk = stream.read(4096)
            if not chunk:
                break
            remaining = _PROCESS_OUTPUT_LIMIT - captured
            if remaining > 0:
                retained = chunk[:remaining]
                output.append(retained)
                captured += len(retained)
                compact = " ".join(retained[:2000].split())
                if compact:
                    log(f"{prefix} {compact[:500]}")
            if len(chunk) > remaining and not truncated:
                truncated = True
                log(f"{prefix} output truncated at {_PROCESS_OUTPUT_LIMIT} characters")
    finally:
        stream.close()


def _run_workflow_runner(
    handoff_path: Path,
    result_path: Path,
    *,
    timeout_seconds: int,
    logger: logging.Logger,
) -> str:
    """Run workflow_runner.py in its own session and stream its output.

    The runner keeps the Claude SDK client alive until GEAK's background Workflow
    completes; ``start_new_session=True`` lets us tear down the whole group
    (runner + claude + any background Workflow child) on timeout or error.
    """
    runner = Path(__file__).with_name("workflow_runner.py")
    command = [sys.executable, str(runner), str(handoff_path), str(result_path)]
    process = subprocess.Popen(
        command,
        cwd=str(handoff_path.parent),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        bufsize=1,
        start_new_session=True,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    stdout: list[str] = []
    stderr: list[str] = []
    stdout_thread = threading.Thread(
        target=_stream_pipe,
        args=(process.stdout, "[GEAK]", stdout, logger.info),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_stream_pipe,
        args=(process.stderr, "[GEAK STDERR]", stderr, logger.warning),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()

    timed_out = False
    try:
        process.wait(timeout=timeout_seconds + 30)
    except subprocess.TimeoutExpired:
        timed_out = True
        logger.error("GEAK v4 runner exceeded its hard timeout")
    finally:
        _terminate_process_group(process, logger)
        stdout_thread.join(timeout=2)
        stderr_thread.join(timeout=2)

    if timed_out:
        raise TimeoutError(f"GEAK v4 timed out after {timeout_seconds} seconds")

    result = _read_json(result_path)
    stderr_text = "".join(stderr)
    if process.returncode != 0:
        detail = result.get("error") if result else stderr_text[-4000:]
        raise RuntimeError(
            f"GEAK v4 runner failed with exit {process.returncode}: {detail}"
        )
    if result is None:
        raise RuntimeError(f"GEAK v4 runner did not write a valid result: {result_path}")
    return "\n".join(part for part in ("".join(stdout), stderr_text) if part)


@register_agent("geak_v4")
def launch_agent(
    eval_config: dict[str, Any],
    task_config_dir: str,
    workspace: str,
) -> str:
    """Run GEAK v4 against the Arena workspace, applying its patch in place."""
    logger = logging.getLogger(__name__)
    agent_config = _load_yaml(Path(__file__).with_name("agent_config.yaml"))
    task_config = _load_yaml(task_config_dir)
    workspace_path = Path(workspace).resolve()
    if not workspace_path.is_dir():
        raise FileNotFoundError(f"Arena workspace does not exist: {workspace_path}")

    task_type = str(task_config.get("task_type") or "")
    supported = {str(value) for value in agent_config.get("supported_task_types", [])}
    if task_type not in supported:
        raise ValueError(
            f"GEAK v4 does not support task_type={task_type!r}; "
            f"supported task types: {sorted(supported)}"
        )
    sources = _declared_sources(task_config, workspace_path)

    workflow_dir = Path(
        os.environ.get("GEAK_V4_WORKFLOW_DIR")
        or agent_config.get("workflow_dir")
        or "/opt/geak/kernel_workflow"
    ).resolve()
    workflow_script = workflow_dir / "kernel_workflow.js"
    if not workflow_script.is_file():
        raise FileNotFoundError(
            f"GEAK v4 workflow not found: {workflow_script}. "
            "Point GEAK_V4_WORKFLOW_DIR at your GEAK kernel_workflow directory."
        )
    claude_binary = shutil.which("claude")
    if not claude_binary:
        raise RuntimeError(
            "Claude Code CLI ('claude') not found on PATH; install it and log in first."
        )

    # Run artifacts live OUTSIDE the workspace: GEAK copies kernel_path into its
    # own exp tree, so nesting outputs under the workspace would recurse — and it
    # keeps GEAK's scratch clear of the directory Arena scores.
    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        + f"_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    )
    run_dir = workspace_path.parent / f".{workspace_path.name}_geak_v4" / run_id
    eval_dir = run_dir / "eval"
    exp_root = run_dir / "runs"
    handoff_path = run_dir / "handoff.json"
    result_path = run_dir / "result.json"
    run_dir.mkdir(parents=True, exist_ok=True)

    prompt_builder = load_prompt_builder(AgentType.GEAK_V4, logger)
    task_prompt = prompt_builder(task_config_dir, str(workspace_path), eval_config, logger)
    if sources:
        joined = ", ".join(f"`{name}`" for name in sources)
        task_prompt += (
            "\n\n### GEAK/Arena Integration Contract\n"
            "The task's config.yaml compile, correctness, and performance commands "
            "are the measurement source of truth. Do not create, modify, or replace "
            "any test, harness, config, reference, or timing file. Optimize only the "
            f"declared source file(s): {joined}."
        )

    timeout_seconds = int(agent_config.get("timeout_seconds", 43200))
    handoff = {
        "schema_version": 1,
        "kernel_path": str(workspace_path),
        "workflow_dir": str(workflow_dir),
        "eval_dir": str(eval_dir),
        "exp_root": str(exp_root),
        "gpu_ids": _logical_gpu_ids(eval_config),
        "budget": int(agent_config.get("budget", 6)),
        "min_improve": float(agent_config.get("min_improve", 0.02)),
        "deep_cost": int(agent_config.get("deep_cost", 2)),
        "use_expert_skills": bool(agent_config.get("use_expert_skills", False)),
        "task": task_prompt,
        "model": str(agent_config.get("model", "claude-opus-4-8")),
        "effort": str(agent_config.get("effort", "ultracode")),
        "claude_cli_path": claude_binary,
        "timeout_seconds": timeout_seconds,
        "done_grace_seconds": float(agent_config.get("done_grace_seconds", 1800)),
        "done_poll_seconds": float(agent_config.get("done_poll_seconds", 5)),
        # GEAK's Director git-applies the validated patch straight into kernel_path
        # (the Arena workspace). Arena's harness guard + independent re-score are
        # the integrity boundary, so we let GEAK edit in place like forge does.
        "apply_to_original": "true",
    }
    handoff_path.write_text(
        json.dumps(handoff, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    logger.info("GEAK v4 preflight")
    logger.info("  workflow: %s", workflow_script)
    logger.info("  workspace (kernel_path): %s", workspace_path)
    logger.info("  eval dir: %s", eval_dir)
    logger.info("  optimize-only sources: %s", sources or "<none declared>")
    logger.info("  logical GPU IDs: %s", handoff["gpu_ids"])
    logger.info("  budget: %s  timeout: %ss", handoff["budget"], timeout_seconds)

    output = _run_workflow_runner(
        handoff_path,
        result_path,
        timeout_seconds=timeout_seconds,
        logger=logger,
    )
    result = _read_json(result_path)
    if result is None:
        raise RuntimeError(f"GEAK v4 runner did not write a valid result: {result_path}")

    status = str(result.get("status") or "unknown")
    applied = str(result.get("applied_to_original", "unknown")).lower() == "true"
    if status == "ok" and applied:
        logger.info(
            "GEAK v4 accepted a gain (speedup=%s); patch applied into the workspace",
            result.get("final_speedup"),
        )
    elif status in {"no_gain", "rejected"}:
        logger.info(
            "GEAK v4 produced no accepted gain (status=%s); workspace left at baseline",
            status,
        )
    else:
        logger.warning(
            "GEAK v4 finished with status=%s (%s)",
            status,
            result.get("reason") or "",
        )
    return output + "\n" + json.dumps(result, sort_keys=True)
