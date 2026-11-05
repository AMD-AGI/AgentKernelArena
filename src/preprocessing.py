# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Run environment setup and task workspace materialization.

Workspaces are created by ``src.task_materialization`` from a schema-v3
``task.yaml`` or a schema-v2 ``config.yaml``; this module only names them and
decides whether a run already completed them.
"""
import os
import subprocess
import logging
from pathlib import Path

import yaml


def _resolve_gfx_arch(target_gpu_model: str) -> str | None:
    """
    Look up the gfx architecture token (e.g. 'gfx942') for a given GPU model
    name (e.g. 'MI300') from default_cheatsheet.yaml.

    Returns None if the GPU model is not found.
    """
    cheatsheet_path = (
        Path(__file__).resolve().parent / "prompts" / "cheatsheet" / "default_cheatsheet.yaml"
    )
    try:
        config = yaml.safe_load(cheatsheet_path.read_text()) or {}
    except Exception:
        return None

    arch_map = config.get("architecture", {})
    gpu_key = str(target_gpu_model)
    entry = (
        arch_map.get(gpu_key)
        or arch_map.get(gpu_key.upper())
        or arch_map.get(gpu_key.lower())
    )
    if isinstance(entry, dict):
        return entry.get("gfx_arch")
    return None


def _detect_gfx_arch_from_rocminfo() -> str | None:
    """Detect the actual GPU gfx arch from rocminfo (e.g. 'gfx950')."""
    try:
        result = subprocess.run(
            ["rocminfo"], capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                stripped = line.strip()
                if stripped.startswith("Name:") and "gfx" in stripped:
                    arch = stripped.split("Name:")[-1].strip()
                    if arch.startswith("gfx"):
                        return arch
    except Exception:
        pass
    return None


_ROCM_ARCH_ENV_VARS = ("PYTORCH_ROCM_ARCH", "AMDGPU_TARGETS", "GPU_TARGETS")


def setup_rocm_env(target_gpu_model: str, logger: logging.Logger) -> None:
    """
    Set the ROCm GPU-arch environment for correct compilation. Exports
    all three of:
        - ``PYTORCH_ROCM_ARCH``  (PyTorch / torch.utils.cpp_extension)
        - ``AMDGPU_TARGETS``     (CMake / HIP)
        - ``GPU_TARGETS``        (CMake / HIP)

    All three are exported together regardless of how the arch was
    resolved, so CMake-based HIP builds always see the same arch as
    PyTorch — including on the common case where ``rocminfo`` succeeds.

    Resolution priority:
        1. Auto-detect from ``rocminfo`` (most reliable — uses actual
           hardware).
        2. Fall back to cheatsheet lookup from ``target_gpu_model``.
        3. Leave the environment unchanged if neither works.
    """
    detected_arch = _detect_gfx_arch_from_rocminfo()
    if detected_arch:
        gfx_arch = detected_arch
        source = "auto-detected from rocminfo"
    else:
        gfx_arch = _resolve_gfx_arch(target_gpu_model)
        if not gfx_arch:
            logger.warning(
                f"Could not resolve gfx arch for GPU model '{target_gpu_model}'. "
                f"None of {_ROCM_ARCH_ENV_VARS} will be set; PyTorch and CMake "
                "will fall back to their built-in arch lists."
            )
            return
        source = f"from target_gpu_model={target_gpu_model}"

    for var in _ROCM_ARCH_ENV_VARS:
        os.environ[var] = gfx_arch
    logger.info(
        f"Set {', '.join(f'{v}={gfx_arch}' for v in _ROCM_ARCH_ENV_VARS)} ({source})"
    )


def _sanitize_task_name(task_name: str) -> str:
    """Convert a task name like 'gemm/gemm_a8w8/n4096' to 'gemm_gemm_a8w8_n4096' for directory names."""
    return task_name.replace("/", "_")


def get_task_workspace_path(run_directory: Path, task_name: str, timestamp: str) -> Path:
    """Return the expected task workspace path for a run."""
    sanitized = _sanitize_task_name(task_name)
    return run_directory / f"{sanitized}_{timestamp}"


def is_task_complete(
    run_directory: Path,
    task_name: str,
    timestamp: str,
    agent_name: str | None = None,
) -> bool:
    """
    Check if a task is already completed.

    Args:
        run_directory: Run-level directory (e.g., workspace_MI300_cursor/run_20250115_143022/)
        task_name: Stable task name (e.g., "normalization/rms_norm/bf16_h8192")
        timestamp: Timestamp string used in task directory name
        agent_name: qa_auditor uses the framework-finalized QA report.
            Optimization requires the framework completion record.

    Returns:
        True if the task's applicable completion contract is satisfied
    """
    task_dir = get_task_workspace_path(run_directory, task_name, timestamp)
    if agent_name == "qa_auditor":
        # Only a report finalized with its completion marker counts; a file a
        # model wrote is not completion evidence.
        from agents.qa_auditor.report import qa_report_is_complete

        return qa_report_is_complete(task_dir)
    from src.task_run import task_run_is_complete

    return task_run_is_complete(task_dir, expected_task_name=task_name, agent_name=agent_name)


def setup_workspace(task_config_dir: str, run_directory: Path, timestamp: str, logger: logging.Logger,
                    task_name: str = "", target_language: str | None = None) -> Path:
    """Materialize a task workspace through its TaskSpec.

    A schema-v3 ``task.yaml`` resolves for the run's ``target_language``; a
    schema-v2 ``config.yaml`` (the e2e serving tasks) declares its own backend.
    A completed workspace is reused only after its framework-side record is
    verified; candidate edits are retained and setup is not rerun on resume.

    Args:
        task_config_dir: Path to the task's task.yaml or config.yaml
        run_directory: Run-level directory (e.g., workspace_MI300_cursor/run_20250115_143022/)
        timestamp: Timestamp string for unique workspace naming
        logger: Logger instance
        task_name: Stable discovered task name (e.g., "normalization/rms_norm/bf16_h8192")
        target_language: The run's target language for a schema-v3 task

    Returns:
        Path to the created workspace directory
    """
    from src.task_materialization import materialize_task_workspace
    from src.task_spec import TaskConfigError, load_task_spec

    task_config_path = Path(task_config_dir)
    stable_id = task_name
    if not stable_id:
        try:
            stable_id = task_config_path.parent.resolve().relative_to(
                Path(__file__).resolve().parents[1] / "tasks"
            ).as_posix()
        except ValueError as exc:
            raise TaskConfigError("setup_workspace requires the stable discovered task_name") from exc
    spec = load_task_spec(task_config_path, task_id=stable_id, target_language=target_language)
    return materialize_task_workspace(
        spec, task_config_path,
        get_task_workspace_path(run_directory, stable_id, timestamp), logger,
    )
