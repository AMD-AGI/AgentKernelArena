# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
# This script will setup environment tools and dependencies. It will also provide duplicated workspace for the agent
import os
import shutil
import subprocess
import logging
from pathlib import Path

import yaml
from typing import Optional


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


def _extract_repo_name(repo_url: str) -> str:
    """Extract repository name from URL (e.g. 'https://github.com/ROCm/rocPRIM.git' -> 'rocPRIM')."""
    # Remove trailing slashes and .git suffix
    url = repo_url.rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]
    # Extract last path component
    return url.rsplit("/", 1)[-1]


def _ensure_repo_cloned(repo_url: str, target_dir: Path, logger: logging.Logger) -> Path:
    """
    Ensure repo is cloned to target_dir. Skip if already exists.
    
    Args:
        repo_url: Git repository URL
        target_dir: Directory to clone into
        logger: Logger instance
    
    Returns:
        Path to the repository directory
    """
    if (target_dir / ".git").exists():
        logger.info(f"Repository already exists at {target_dir}, skipping clone")
        return target_dir

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Cloning {repo_url} into {target_dir}")
    try:
        subprocess.run(
            ["git", "clone", repo_url, str(target_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"git clone failed: {(e.stderr or '').strip()}") from e

    return target_dir


def setup_repo_from_config(
    task_config_dir: str, workspace_path: Path, logger: logging.Logger
) -> Optional[Path]:
    """Return workspace repo path if task has repo_url, else None."""
    with open(task_config_dir, "r") as f:
        task_config = yaml.safe_load(f) or {}
    repo_url = task_config.get("repo_url")
    if not repo_url:
        return None
    repo_subdir = task_config.get("repo_subdir") or _extract_repo_name(repo_url)
    repo_dir = workspace_path / repo_subdir
    return repo_dir if (repo_dir / ".git").exists() else None


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

    For tasks with repo_url:
      1. Clone repo into tasks/ directory (if not already cloned)
      2. Copy entire task folder (including repo) to workspace

    Args:
        task_config_dir: Path to task's config.yaml
        run_directory: Run-level directory (e.g., workspace_MI300_cursor/run_20250115_143022/)
        timestamp: Timestamp string for unique workspace naming
        logger: Logger instance
        task_name: Full task name (e.g., "hip2hip/gpumode/SiLU") for unique directory naming

    Returns:
        Path to the created workspace directory
    """
    task_config_path = Path(task_config_dir)
    task_folder = task_config_path.parent

    # Load task config
    with open(task_config_path, "r") as f:
        task_config = yaml.safe_load(f) or {}

    # 1. Clone repo into tasks/ directory if needed (only once, reused by all runs)
    repo_url = task_config.get("repo_url")
    if repo_url:
        repo_subdir = task_config.get("repo_subdir") or _extract_repo_name(repo_url)
        repo_in_tasks = task_folder / repo_subdir
        _ensure_repo_cloned(repo_url, repo_in_tasks, logger)

    # 2. Create workspace directory
    if task_name:
        new_folder_name = f"{_sanitize_task_name(task_name)}_{timestamp}"
    else:
        new_folder_name = f"{task_folder.name}_{timestamp}"
    workspace_path = run_directory / new_folder_name
    workspace_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created workspace directory: {workspace_path}")

    # 3. Copy entire task folder (including cloned repo) to workspace
    for item in task_folder.iterdir():
        dst = workspace_path / item.name
        if item.is_dir():
            shutil.copytree(item, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dst)

    logger.info(f"Copied task folder content from {task_folder} to {workspace_path}")

    return workspace_path
