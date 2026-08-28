# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Forge rewrite agent — bridges Arena to `kernel-agents forge-rewrite-by-flydsl`.

The rewrite pipeline reimplements an operator in FlyDSL from its existing
implementation (PORT, correctness only) and then optimizes the port with a
nested forge-loop. This launcher adapts an Arena task to it:

  1. Build a scratch workspace beside the task as an empty git repository with
     an unborn HEAD. KernelForge reads the workspace's HEAD as the base commit
     for framework apply-back and makes apply-back a success requirement when
     one exists; an Arena task has no framework repository to patch, so the
     scratch workspace must report no base commit -- and it needs its own .git
     to stop git from walking up into the Arena workspace's repository.
  2. Copy the port source (the baseline implementation's entry file) and the
     task's dual-path driver into it. The PORT prompt reports the source by
     basename only, so it has to be reachable from the rewrite workspace.
  3. Shell out to the rewrite CLI (streaming output).
  4. Copy the ported FlyDSL kernel back onto the task's declared port target so
     Arena's own compile/correctness/performance commands score it.

The task's ``rewrite:`` block supplies port_source, port_source_entry,
port_target and logical_operator; see
tasks/SIKL-task/glm52_moe_mxfp4_per1x32_t64/config.yaml.
"""

from __future__ import annotations

import json
import logging
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml

from agents import register_agent
from agents.forge.common import (
    _capture_forge_edit_baseline,
    _declared_editable_sources,
    _forge_max_hours,
    _init_git_workspace,
    _read_forge_result,
    _resolve_all_source_files,
    _resolve_gpu_arch,
    _resolve_gpu_type,
    _verify_forge_edit_scope,
    forge_environment,
    run_forge_subprocess,
)

REWRITE_WORKSPACE_DIR = "forge_rewrite_ws"
RESULT_FILE = "forge_rewrite_result.json"
STATUS_FILE = "arena_forge_rewrite_status.json"


def _rewrite_config(task_config: dict[str, Any], task_config_dir: str) -> dict[str, Any]:
    """Read and validate the task's rewrite block."""
    block = task_config.get("rewrite")
    if not isinstance(block, dict):
        raise RuntimeError(
            f"Task config has no 'rewrite' mapping: {task_config_dir}. A "
            "rewrite_by_flydsl task must declare port_source, port_target and "
            "logical_operator."
        )
    missing = [
        key for key in ("port_source", "port_target", "logical_operator")
        if not str(block.get(key) or "").strip()
    ]
    if missing:
        raise RuntimeError(f"Task rewrite block is missing: {', '.join(missing)}")
    return block


def _resolve_port_source(workspace: str, port_source: str) -> Path:
    """Locate the port source, absolute or workspace-relative."""
    candidates = [Path(port_source), Path(workspace) / port_source]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise RuntimeError(
        f"Port source not found: tried {[str(c) for c in candidates]}. It must be "
        "the baseline implementation's entry file."
    )


def _prepare_rewrite_workspace(
    workspace: str,
    port_source: Path,
    port_target_name: str,
    logger: logging.Logger,
) -> tuple[Path, Path, Path]:
    """Create the scratch workspace and return (root, source copy, driver copy).

    The driver's sibling modules travel with it: the task keeps the operator's
    input construction, reference and baseline under ``scripts/`` and the driver
    imports them from its own directory.
    """
    task_driver = Path(workspace) / "scripts" / "forge_driver.py"
    if not task_driver.is_file():
        raise RuntimeError(
            f"A rewrite_by_flydsl task must ship a dual-path driver at {task_driver}"
        )

    root = Path(workspace) / REWRITE_WORKSPACE_DIR
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    # An empty git repository, deliberately without a commit. git discovers a
    # repository by walking up, so a scratch directory with no .git of its own
    # resolves the Arena workspace's HEAD instead, and KernelForge reads that
    # HEAD as the framework base commit -- which makes apply-back a success
    # requirement for a task that has no framework repository to patch. An
    # initialized repository with an unborn HEAD stops the upward walk and
    # reports no base commit, while still letting the pipeline commit the port.
    subprocess.run(
        ["git", "init", "--quiet", str(root)],
        capture_output=True,
        text=True,
        check=True,
    )

    for module in sorted(task_driver.parent.glob("*.py")):
        if module.name == port_target_name:
            # The candidate must be the only importable module of that name:
            # KernelForge rejects a driver whose directory shadows the port.
            logger.warning(
                "forge_rewrite: not copying %s into the rewrite workspace; it "
                "would shadow the FlyDSL candidate", module.name,
            )
            continue
        shutil.copy2(module, root / module.name)

    # Arena materializes its canonical benchmark helper beside every file that
    # imports it, so the loop above carries it in with the driver's modules. The
    # driver times both paths with it, which is what keeps the pipeline's own
    # speedup and the task's score in one measurement regime.
    if not (root / "_aka_benchmark.py").is_file():
        raise RuntimeError(
            f"Canonical benchmark helper not found in {task_driver.parent}; the "
            "rewrite driver needs it to time the baseline and the candidate the "
            "same way Arena scores them. Does the driver import _aka_benchmark?"
        )

    source_copy = root / port_source.name
    if source_copy.name == port_target_name:
        raise RuntimeError(
            "The port source and the port target cannot share a file name: "
            f"{port_target_name}"
        )
    shutil.copy2(port_source, source_copy)

    driver_copy = root / task_driver.name
    logger.info(f"forge_rewrite: rewrite workspace {root}")
    logger.info(f"forge_rewrite:   port source  {source_copy}")
    logger.info(f"forge_rewrite:   driver       {driver_copy}")
    return root, source_copy, driver_copy


def _build_rewrite_command(
    *,
    forge_bin: str,
    rewrite_root: Path,
    source_copy: Path,
    driver_copy: Path,
    result_json: Path,
    rewrite: dict[str, Any],
    agent_config: dict[str, Any],
    gpu_arch: str,
    gpu_type: str,
) -> list[str]:
    """Build argv without shell parsing so task metadata is forwarded exactly."""
    port_target = Path(str(rewrite["port_target"])).name
    source_entry = str(rewrite.get("port_source_entry") or "").strip()
    snr_threshold = rewrite.get("snr_threshold", agent_config.get("snr_threshold", 30.0))
    max_port_attempts = int(
        rewrite.get("max_port_attempts", agent_config.get("max_port_attempts", 3))
    )

    cmd = [
        forge_bin,
        "forge-rewrite-by-flydsl",
        "--source-kernel",
        str(source_copy),
        "--driver",
        str(driver_copy),
        "--no-prepare-driver",
        "--logical-op-name",
        str(rewrite["logical_operator"]),
        "--workspace",
        str(rewrite_root),
        "--experiments-dir",
        str(rewrite_root / "forge_experiments"),
        "--result-json",
        str(result_json),
        "--flydsl-kernel-name",
        port_target,
        "--gpu-target",
        gpu_arch,
        "--gpu-type",
        gpu_type,
        "--snr-threshold",
        str(snr_threshold),
        "--max-port-attempts",
        str(max_port_attempts),
        "--max-hours",
        str(_forge_max_hours(agent_config)),
        "--model",
        str(agent_config.get("model", "claude-opus-4-8")),
        "--permission-mode",
        str(agent_config.get("permission_mode", "acceptEdits")),
        "--supervisor-backend",
        str(agent_config.get("supervisor_backend", "codex")),
        "--rewrite-kb" if agent_config.get("rewrite_kb") else "--no-rewrite-kb",
    ]
    if source_entry:
        cmd.extend(["--source-entry", source_entry, "--target-functions", source_entry])
    source_language = str(rewrite.get("port_source_language") or "").strip()
    if source_language:
        cmd.extend(["--source-language", source_language])
    shapes = rewrite.get("shapes")
    if shapes:
        cmd.extend(["--shapes-json", json.dumps(shapes)])
    return cmd


def _locate_ported_kernel(
    rewrite_root: Path,
    result: dict[str, Any] | None,
    port_target_name: str,
) -> Path | None:
    """Find the FlyDSL kernel the pipeline produced, newest attempt first."""
    if result:
        for relative in result.get("temporary_paths") or []:
            candidate = rewrite_root / str(relative) / port_target_name
            if candidate.is_file():
                return candidate
    attempts = sorted(
        (rewrite_root / ".forge_rewrite").glob(f"*/{port_target_name}"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return attempts[0] if attempts else None


def _write_rewrite_status(
    rewrite_root: Path,
    *,
    returncode: int | None,
    timed_out: bool,
    result: dict[str, Any] | None,
    ported_kernel: Path | None,
) -> dict[str, Any]:
    summary = {
        "exit_code": returncode,
        "timed_out": timed_out,
        "port_ok": bool((result or {}).get("port_ok", False)),
        "success": bool((result or {}).get("success", False)),
        "failure_class": (result or {}).get("failure_class", ""),
        "failure_detail": (result or {}).get("failure_detail", ""),
        "source_ms": (result or {}).get("source_ms"),
        "flydsl_best_ms": (result or {}).get("flydsl_best_ms"),
        "speedup": (result or {}).get("speedup"),
        "builder_symbol": (result or {}).get("builder_symbol", ""),
        "ported_kernel": str(ported_kernel) if ported_kernel else "",
    }
    experiments_dir = rewrite_root / "forge_experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)
    (experiments_dir / STATUS_FILE).write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    return summary


@register_agent("forge_rewrite")
def launch_agent(eval_config: dict[str, Any], task_config_dir: str, workspace: str) -> str:
    """Run one KernelForge rewrite-by-flydsl campaign over an Arena task.

    Args:
        eval_config: Arena run config (includes target_gpu_model).
        task_config_dir: Path to the task's config.yaml.
        workspace: Isolated task workspace Arena prepared.

    Returns:
        Combined streamed output of the rewrite subprocess.
    """
    logger = logging.getLogger(__name__)

    forge_bin = shutil.which("kernel-agents")
    if not forge_bin:
        raise RuntimeError(
            "Command 'kernel-agents' not found. Install KernelForge "
            "(pip install -e KernelForge) so the rewrite CLI is on PATH."
        )

    config_path = Path(__file__).with_name("agent_config.yaml")
    with config_path.open("r") as f:
        agent_config = yaml.safe_load(f) or {}

    with open(task_config_dir, "r") as f:
        task_config = yaml.safe_load(f) or {}
    rewrite = _rewrite_config(task_config, task_config_dir)
    port_target_name = Path(str(rewrite["port_target"])).name
    port_source = _resolve_port_source(workspace, str(rewrite["port_source"]))

    editable_sources = _resolve_all_source_files(
        workspace,
        _declared_editable_sources(task_config),
        task_config,
        logger,
    )

    gpu_arch = _resolve_gpu_arch(eval_config)
    gpu_type = _resolve_gpu_type(eval_config)
    env = forge_environment()

    # Arena's own edit boundary is enforced against this commit. The rewrite
    # scratch directory is gitignored, so it neither dirties the tree nor gets
    # discarded as undeclared scratch.
    _init_git_workspace(workspace, logger)
    edit_baseline = _capture_forge_edit_baseline(workspace)

    rewrite_root, source_copy, driver_copy = _prepare_rewrite_workspace(
        workspace, port_source, port_target_name, logger
    )
    result_json = rewrite_root / "forge_experiments" / RESULT_FILE

    cmd_parts = _build_rewrite_command(
        forge_bin=forge_bin,
        rewrite_root=rewrite_root,
        source_copy=source_copy,
        driver_copy=driver_copy,
        result_json=result_json,
        rewrite=rewrite,
        agent_config=agent_config,
        gpu_arch=gpu_arch,
        gpu_type=gpu_type,
    )

    logger.info("Forge Rewrite Preflight")
    logger.info(f"  forge bin:   {forge_bin}")
    logger.info(f"  port source: {source_copy} (from {port_source})")
    logger.info(f"  port target: {port_target_name}")
    logger.info(f"  driver:      {driver_copy}")
    logger.info(f"  operator:    {rewrite['logical_operator']}")
    logger.info(f"  gpu target:  {gpu_arch}")
    logger.info(f"  gpu type:    {gpu_type}")
    logger.info(f"  model:       {agent_config.get('model')}")
    logger.info(f"  budget:      {_forge_max_hours(agent_config)}h")
    logger.info(f"  gateway:     {env.get('ANTHROPIC_BASE_URL', '<unset>')}")
    logger.info(f"Running command: {' '.join(shlex.quote(p) for p in cmd_parts)}")
    logger.info("=" * 80)
    logger.info("Forge Rewrite Output (streaming):")
    logger.info("=" * 80)

    timeout_seconds = int(agent_config.get("timeout_seconds", 3600))
    process, stdout_lines, stderr_lines, timed_out = run_forge_subprocess(
        cmd_parts,
        workspace=str(rewrite_root),
        env=env,
        timeout_seconds=timeout_seconds,
        logger=logger,
    )

    logger.info("=" * 80)
    logger.info(f"Forge rewrite completed with exit code: {process.returncode}")
    logger.info("=" * 80)

    result = _read_forge_result(result_json, "\n".join(stdout_lines))
    ported_kernel = _locate_ported_kernel(rewrite_root, result, port_target_name)
    status = _write_rewrite_status(
        rewrite_root,
        returncode=process.returncode,
        timed_out=timed_out,
        result=result,
        ported_kernel=ported_kernel,
    )

    output = "\n".join(stdout_lines)
    if stderr_lines:
        output += "\n=== STDERR ===\n" + "\n".join(stderr_lines)

    if not status["port_ok"] or ported_kernel is None:
        # Refuse to fall through to scoring: Arena would re-measure the baseline
        # through the task's stub path and report it as an unimproved result,
        # which reads as "the port worked and was not faster".
        raise RuntimeError(
            "forge_rewrite produced no FlyDSL port "
            f"(port_ok={status['port_ok']}, failure_class="
            f"{status['failure_class'] or '<none>'}, detail="
            f"{status['failure_detail'] or '<none>'})"
        )

    destination = Path(workspace) / port_target_name
    shutil.copy2(ported_kernel, destination)
    logger.info(f"forge_rewrite: installed ported kernel -> {destination}")
    logger.info(
        "forge_rewrite result: source_ms=%s flydsl_best_ms=%s speedup=%s",
        status["source_ms"],
        status["flydsl_best_ms"],
        status["speedup"],
    )

    # No `git checkout` here: unlike the forge-loop path, nothing edits the Arena
    # workspace tree during the run, and the ported kernel installed above is an
    # uncommitted change a checkout would discard.
    _verify_forge_edit_scope(workspace, edit_baseline, editable_sources, logger)
    return output
