# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""operator2flydsl via KernelForge's `kernel-agents forge-rewrite-by-flydsl`.

This is one provider of the ``operator2flydsl`` task type, not the definition of
it: the task says which operator to reimplement in FlyDSL and where it is scored,
and this agent adapts that to KernelForge's CLI. A different provider would read
the same task and never mention a port attempt or a rewrite pipeline.

That pipeline reimplements an operator in FlyDSL from its existing
implementation (PORT, correctness only) and then optimizes the port with a
nested forge-loop. This launcher adapts an Arena task to it:

  1. Build a scratch workspace beside the task as its own git repository with
     one commit. KernelForge's agent sessions require a git worktree with a
     resolvable HEAD, and it must be the scratch directory's own repository so
     that the framework base commit -- and therefore any apply-back patch -- is
     never computed against the Arena task's files.

     An Arena task has no framework repository to patch, so KernelForge reports
     ``applyback_required: true`` and ``success: false`` for an otherwise good
     port. This launcher therefore gates on ``port_ok``, and reserves nothing:
     the pipeline itself keeps its last 20 minutes for the apply-back stage.
  2. Copy the port source (the baseline implementation's entry file) and the
     task's dual-path driver into it. The PORT prompt reports the source by
     basename only, so it has to be reachable from the rewrite workspace.
  3. Shell out to the rewrite CLI (streaming output).
  4. Copy the ported FlyDSL kernel back onto the task's single declared editable
     source so Arena's own compile/correctness/performance commands score it.

Everything it reads from the task is a structured field, and all but one of them
already existed: ``source_file_path[0]`` is where the port lands,
``kernel_identity.logical_operator`` is the operator's identity, and
``rewrite_source_file`` -- the one field this task type adds -- names the
production implementation to port from. How the campaign searches (attempt
count, coarse filter, budget, model) is agent configuration and is never read
from a task. See tasks/SIKL-task/gemm_a16w16_nt_n6144_k6144/config.yaml.
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
    _logical_operator,
    _read_forge_result,
    _resolve_all_source_files,
    _resolve_gpu_arch,
    _resolve_gpu_type,
    _verify_forge_edit_scope,
    forge_environment,
    run_forge_subprocess,
)

REWRITE_WORKSPACE_DIR = "forge_operator2flydsl_ws"
RESULT_FILE = "forge_operator2flydsl_result.json"
STATUS_FILE = "arena_forge_operator2flydsl_status.json"

# Everything KernelForge itself creates inside the scratch repository while it
# runs. The agent-session guard rejects a session that leaves new non-ignored
# files behind, and the pipeline writes the seeded candidate, its bytecode cache
# and its experiment tree into the workspace, so ignoring them is what lets a
# port session end at all. The candidate is force-added when the pipeline
# commits it, so ignoring .forge_rewrite/ does not hide the port.
_SCRATCH_GITIGNORE = """\
__pycache__/
*.pyc
*.log
build/
forge_experiments/
.forge_rewrite/
"""


def _port_target(task_config: dict[str, Any], task_config_dir: str) -> str:
    """The file the port lands in: the task's single declared editable source."""
    declared = task_config.get("source_file_path") or []
    if not isinstance(declared, list) or not declared:
        raise RuntimeError(
            f"Task config declares no source_file_path: {task_config_dir}. An "
            "operator2flydsl task makes exactly one file editable, and the port "
            "lands in it."
        )
    return Path(str(declared[0])).name


def _resolve_source_file(
    workspace: str, task_config: dict[str, Any], task_config_dir: str
) -> Path:
    """Locate the production implementation the operator is ported from.

    Task-relative first, which is where it belongs: an absolute path binds the
    task to one image layout and defeats Arena's isolated-workspace contract.
    Absolute paths still resolve while the SIKL sources come from the runtime
    image, because nothing materializes them into the workspace yet.
    """
    declared = str(task_config.get("rewrite_source_file") or "").strip()
    if not declared:
        raise RuntimeError(
            f"Task config declares no rewrite_source_file: {task_config_dir}. An "
            "operator2flydsl task must name the production implementation it "
            "ports from."
        )
    candidates = [Path(workspace) / declared, Path(declared)]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise RuntimeError(
        f"rewrite_source_file not found: tried {[str(c) for c in candidates]}"
    )


def _init_scratch_repository(root: Path) -> None:
    """Make the scratch workspace its own git repository with one commit.

    KernelForge's agent sessions run behind a workspace guard that snapshots the
    repository to roll a session back, and for a writable session it requires a
    git worktree with a resolvable HEAD: a plain directory fails its
    ``rev-parse --show-toplevel`` and a repository with an unborn HEAD fails the
    HEAD snapshot, either way ending every port attempt as an "agent session
    error" before the agent does any work.

    It has to be the scratch directory's OWN repository rather than the Arena
    workspace's: git resolves a repository by walking up, so otherwise the
    pipeline would read the Arena workspace's HEAD as the framework base commit
    and any apply-back patch would be computed against the task's own files.

    The same guard rejects a session that leaves new non-ignored files behind,
    which is what the .gitignore is for: the pipeline writes the seeded
    candidate and its experiment tree into this repository while the session
    runs.
    """
    (root / ".gitignore").write_text(_SCRATCH_GITIGNORE)
    commands = (
        ["git", "init", "--quiet", "."],
        ["git", "config", "user.email", "forge-operator2flydsl@local"],
        ["git", "config", "user.name", "forge-operator2flydsl"],
        ["git", "add", "-A"],
        ["git", "commit", "--quiet", "-m", "forge-operator2flydsl: scratch workspace"],
    )
    for command in commands:
        subprocess.run(command, cwd=root, capture_output=True, text=True, check=True)


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
            f"An operator2flydsl task must ship a dual-path driver at {task_driver}"
        )

    root = Path(workspace) / REWRITE_WORKSPACE_DIR
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    for module in sorted(task_driver.parent.glob("*.py")):
        if module.name == port_target_name:
            # The candidate must be the only importable module of that name:
            # KernelForge rejects a driver whose directory shadows the port.
            logger.warning(
                "forge_operator2flydsl: not copying %s into the rewrite workspace; it "
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

    # Committed last, so the agent session starts from a clean worktree whose
    # HEAD already contains the driver, its modules and the port source.
    _init_scratch_repository(root)

    driver_copy = root / task_driver.name
    logger.info(f"forge_operator2flydsl: rewrite workspace {root}")
    logger.info(f"forge_operator2flydsl:   port source  {source_copy}")
    logger.info(f"forge_operator2flydsl:   driver       {driver_copy}")
    return root, source_copy, driver_copy


def _build_rewrite_command(
    *,
    forge_bin: str,
    rewrite_root: Path,
    source_copy: Path,
    driver_copy: Path,
    result_json: Path,
    port_target: str,
    source_entry: str,
    logical_operator: str,
    agent_config: dict[str, Any],
    gpu_arch: str,
    gpu_type: str,
) -> list[str]:
    """Build argv without shell parsing so task metadata is forwarded exactly.

    Every search control comes from the agent config, never from the task. How
    many correctness-only attempts a campaign is worth and where its coarse
    filter sits describe how this agent searches, not what the operator computes
    or how Arena scores it; another agent implementing operator2flydsl may have
    no notion of either.

    No ``--rewrite-kb`` / ``--no-rewrite-kb`` is passed, on purpose. Whether the
    recipe store is reachable is a property of the environment rather than of
    the task or the agent, and KernelForge already resolves it from
    KNOWLEDGE_STORE_MODE and the KB_STORE credentials: unset means a local
    store that needs no credentials, ``remote`` means a remote one and names the
    missing variable if it is not configured. Forwarding a flag from Arena's own
    config could only override that with a worse-informed answer -- as it did
    while it defaulted the store off in environments that had it available.

    No ``--source-language`` either: KernelForge infers it from the source file,
    and this launcher has no better information. Guessing it from the suffix
    would be worse than silence, since these tasks port from a Python dispatch
    rather than from a Triton kernel.
    """
    snr_threshold = agent_config.get("snr_threshold", 30.0)
    max_port_attempts = int(agent_config.get("max_port_attempts", 3))

    cmd = [
        forge_bin,
        "forge-rewrite-by-flydsl",
        "--source-kernel",
        str(source_copy),
        "--driver",
        str(driver_copy),
        "--no-prepare-driver",
        "--logical-op-name",
        logical_operator,
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
    ]
    if source_entry:
        cmd.extend(["--source-entry", source_entry, "--target-functions", source_entry])
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


@register_agent("forge_operator2flydsl")
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
    port_target_name = _port_target(task_config, task_config_dir)
    port_source = _resolve_source_file(workspace, task_config, task_config_dir)
    logical_operator = _logical_operator(task_config)
    if not logical_operator:
        raise RuntimeError(
            f"Task config declares no kernel_identity.logical_operator: "
            f"{task_config_dir}. It is the operator's KB identity and what "
            "KernelForge derives the builder symbol from."
        )
    source_entry = str(task_config.get("rewrite_source_entry") or "").strip()

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
        port_target=port_target_name,
        source_entry=source_entry,
        logical_operator=logical_operator,
        agent_config=agent_config,
        gpu_arch=gpu_arch,
        gpu_type=gpu_type,
    )

    logger.info("Forge operator2flydsl Preflight")
    logger.info(f"  forge bin:   {forge_bin}")
    logger.info(f"  port source: {source_copy} (from {port_source})")
    logger.info(f"  port target: {port_target_name}")
    logger.info(f"  driver:      {driver_copy}")
    logger.info(f"  operator:    {logical_operator}")
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
            "forge_operator2flydsl produced no FlyDSL port "
            f"(port_ok={status['port_ok']}, failure_class="
            f"{status['failure_class'] or '<none>'}, detail="
            f"{status['failure_detail'] or '<none>'})"
        )

    destination = Path(workspace) / port_target_name
    shutil.copy2(ported_kernel, destination)
    logger.info(f"forge_operator2flydsl: installed ported kernel -> {destination}")
    logger.info(
        "forge_operator2flydsl result: source_ms=%s flydsl_best_ms=%s speedup=%s",
        status["source_ms"],
        status["flydsl_best_ms"],
        status["speedup"],
    )

    # No `git checkout` here: unlike the forge-loop path, nothing edits the Arena
    # workspace tree during the run, and the ported kernel installed above is an
    # uncommitted change a checkout would discard.
    #
    # That same property is why this path still checks the edit scope at all
    # where forge-loop no longer can. This pipeline runs with
    # --no-prepare-driver inside its own gitignored scratch repository, so it
    # authors no scaffolding here and Arena's pre-launch snapshot still
    # describes exactly what the agent was given.
    undeclared_edits = _verify_forge_edit_scope(
        workspace, edit_baseline, editable_sources, logger
    )
    if undeclared_edits:
        # Carried into the scored output so a reader of the report can tell that
        # this result rests partly on files the task did not declare editable.
        output += "\n=== UNDECLARED EDITS ===\n" + "\n".join(undeclared_edits)
    return output
