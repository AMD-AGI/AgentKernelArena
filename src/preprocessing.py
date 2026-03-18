# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
# This script will setup environment tools and dependencies. It will also provide duplicated workspace for the agent
import os
import shutil
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


def setup_rocm_env(target_gpu_model: str, logger: logging.Logger) -> None:
    """
    Set PYTORCH_ROCM_ARCH (and related env vars) based on config.yaml's
    target_gpu_model so that torch.utils.cpp_extension.load() and hipcc
    compile for the correct GPU architecture.

    Should be called once at the start of main(), before any task is launched.
    """
    gfx_arch = _resolve_gfx_arch(target_gpu_model)
    if not gfx_arch:
        logger.warning(
            f"Could not resolve gfx arch for GPU model '{target_gpu_model}'. "
            "PYTORCH_ROCM_ARCH will not be set; PyTorch will fall back to its built-in arch list."
        )
        return

    os.environ["PYTORCH_ROCM_ARCH"] = gfx_arch
    logger.info(f"Set PYTORCH_ROCM_ARCH={gfx_arch} (from target_gpu_model={target_gpu_model})")


def check_environment() -> None:
    # check hipcc, rocprof-compute
    if "hipcc" not in os.environ["PATH"]:
        raise ValueError("hipcc is not in the PATH")
    if "rocprof-compute" not in os.environ["PATH"]:
        raise ValueError("rocprof-compute is not in the PATH")
    pass


def _sanitize_task_name(task_name: str) -> str:
    """Convert a task name like 'hip2hip/gpumode/SiLU' to 'hip2hip_gpumode_SiLU' for use in directory names."""
    return task_name.replace("/", "_")


def is_task_complete(run_directory: Path, task_name: str, timestamp: str) -> bool:
    """
    Check if a task is already completed.

    Args:
        run_directory: Run-level directory (e.g., workspace_MI300_cursor/run_20250115_143022/)
        task_name: Full task name (e.g., "hip2hip/gpumode/SiLU")
        timestamp: Timestamp string used in task directory name

    Returns:
        True if task directory exists and task_result.yaml exists, False otherwise
    """
    sanitized = _sanitize_task_name(task_name)
    task_dir = run_directory / f"{sanitized}_{timestamp}"
    result_file = task_dir / "task_result.yaml"
    return result_file.exists()


def setup_workspace(task_config_dir: str, run_directory: Path, timestamp: str, logger: logging.Logger,
                    task_name: str = "") -> Path:
    """
    Setup workspace for agent execution by duplicating task directory.

    Args:
        task_config_dir: Path to task's config.yaml
        run_directory: Run-level directory (e.g., workspace_MI300_cursor/run_20250115_143022/)
        timestamp: Timestamp string for unique workspace naming
        logger: Logger instance
        task_name: Full task name (e.g., "hip2hip/gpumode/SiLU") for unique directory naming

    Returns:
        Path to the created workspace directory
    """
    # 1. Get task_folder name (parent directory of task_config_dir)
    task_config_path = Path(task_config_dir)
    task_folder = task_config_path.parent

    # 2. Create new directory with timestamp suffix under run_directory
    # Use sanitized full task_name to avoid collisions between tasks with the same leaf name
    if task_name:
        new_folder_name = f"{_sanitize_task_name(task_name)}_{timestamp}"
    else:
        new_folder_name = f"{task_folder.name}_{timestamp}"
    workspace_path = run_directory / new_folder_name
    workspace_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Created workspace directory: {workspace_path}")

    # 3. Duplicate all content under task_folder to the new workspace folder
    for item in task_folder.iterdir():
        src = item
        dst = workspace_path / item.name
        if item.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)

    logger.info(f"Copied task folder content from {task_folder} to {workspace_path}")

    return workspace_path
