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


def _clone_repo_to_workspace(
    repo_url: str, workspace_path: Path, logger: logging.Logger, subdir_name: Optional[str] = None
) -> Path:
    """
    Clone repo into a subdirectory under workspace (not tasks/ folder).

    This keeps tasks/ directory clean (only config + scripts) and clones
    fresh repo into each workspace.

    Args:
        repo_url: Git repository URL
        workspace_path: Workspace directory (e.g. workspace_MI308_cursor/block_histogram_20260305_...)
        logger: Logger instance
        subdir_name: Optional subdirectory name; if None, extracted from repo_url

    Returns:
        Path to the cloned repository subdirectory
    """
    if subdir_name is None:
        subdir_name = _extract_repo_name(repo_url)

    repo_dir = workspace_path / subdir_name

    # Skip if already cloned (shouldn't happen for fresh workspace, but be safe)
    if (repo_dir / ".git").exists():
        logger.info(f"Repository already exists at {repo_dir}, skipping clone")
        return repo_dir

    logger.info(f"Cloning {repo_url} into {repo_dir}")
    try:
        subprocess.run(
            ["git", "clone", repo_url, str(repo_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"git clone failed: {(e.stderr or '').strip()}") from e

    return repo_dir


def setup_repo_from_config(
    task_config_dir: str, workspace_path: Path, logger: logging.Logger
) -> Optional[Path]:
    """Return workspace repo path if task has repo_url, else None."""
    with open(task_config_dir, "r") as f:
        task_config = yaml.safe_load(f) or {}
    repo_url = task_config.get("repo_url")
    if not repo_url:
        return None
    subdir_name = task_config.get("repo_subdir") or _extract_repo_name(repo_url)
    repo_dir = workspace_path / subdir_name
    return repo_dir if (repo_dir / ".git").exists() else None


def setup_workspace(task_config_dir: str, workspace_directory: str, timestamp: str, logger: logging.Logger) -> Path:
    """
    Setup workspace for agent execution by duplicating task directory.

    For tasks with repo_url:
      1. Copy task files (config.yaml, scripts/, etc.) to workspace
      2. Clone repo into workspace subdirectory (e.g. workspace/rocPRIM/)
    
    This keeps tasks/ directory clean and gives each run a fresh repo clone.

    Args:
        task_config_dir: Path to task's config.yaml
        workspace_directory: Base workspace directory
        timestamp: Timestamp string for unique workspace naming
        logger: Logger instance

    Returns:
        Path to the created workspace directory
    """
    # 1. Get task_folder name (parent directory of task_config_dir)
    task_config_path = Path(task_config_dir)
    task_folder = task_config_path.parent
    task_folder_name = task_folder.name

    # Load task config
    with open(task_config_path, "r") as f:
        task_config = yaml.safe_load(f) or {}

    # 2. Create new directory with timestamp suffix under workspace_dir
    new_folder_name = f"{task_folder_name}_{timestamp}"
    workspace_path = Path(workspace_directory) / new_folder_name
    workspace_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Created workspace directory: {workspace_path}")

    # 3. Copy task folder content to workspace (excluding any previously cloned repos)
    repo_subdir = None
    if repo_url := task_config.get("repo_url"):
        repo_subdir = task_config.get("repo_subdir") or _extract_repo_name(repo_url)

    for item in task_folder.iterdir():
        # Skip repo subdirectory if it exists in task folder (legacy cleanup)
        if repo_subdir and item.name == repo_subdir:
            continue
        src = item
        dst = workspace_path / item.name
        if item.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)

    logger.info(f"Copied task folder content from {task_folder} to {workspace_path}")

    # 4. Clone repo into workspace subdirectory (not tasks/ folder)
    if repo_url:
        _clone_repo_to_workspace(repo_url, workspace_path, logger, repo_subdir)

    return workspace_path
