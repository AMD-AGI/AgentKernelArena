# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
# utils/registry.py

from pathlib import Path
from typing import Dict, Optional


def get_task_config(tasks_root: str = "tasks", category: Optional[str] = None) -> Dict[str, str]:
    """
    Automatically scan all task folders under the tasks directory.
    If config.yaml exists, register it to task_config_list.

    Task naming strategy:
    - Uses relative path from tasks_root as the unique task name
    - Example: tasks/hip2hip/gpumode/SiLU/config.yaml
      -> task_name: "hip2hip/gpumode/SiLU"
    - This ensures uniqueness even when folder names collide

    Args:
        tasks_root: Root directory for tasks (default: "tasks")
        category: Optional category to filter tasks (e.g., "hip2hip", "triton2triton")

    Returns:
        dict: {task_name: config_path}
            - task_name: Relative path from tasks_root to the task folder
            - config_path: Relative path to config.yaml from project root
    """
    task_config_dict = {}
    root = Path(tasks_root)

    # If category is specified, only scan that category
    if category:
        pattern = f"{category}/**/config.yaml"
    else:
        pattern = "**/config.yaml"

    for config_path in root.glob(pattern):
        # Get task name as relative path from tasks_root to the task folder
        # This ensures uniqueness across different categories
        # Example: hip2hip/gpumode/SiLU
        task_folder = config_path.parent
        task_name = str(task_folder.relative_to(root))

        # Store relative path from project root to config.yaml
        relative_path = str(config_path)

        # Register to task_config_dict
        task_config_dict[task_name] = relative_path

    return task_config_dict


