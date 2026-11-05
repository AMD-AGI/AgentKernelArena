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
    - Example: tasks/triton2triton/vllm/triton_rms_norm/config.yaml
      -> task_name: "triton2triton/vllm/triton_rms_norm"
    - This ensures uniqueness even when folder names collide

    Args:
        tasks_root: Root directory for tasks (default: "tasks")
        category: Optional path prefix to filter tasks (e.g., "hip2hip" or
            "repository/rocprim")

    Returns:
        dict: {task_name: config_path}
            - task_name: Relative path from tasks_root to the task folder
            - config_path: Relative path to config.yaml from project root
    """
    task_config_dict = {}
    root = Path(tasks_root)
    prefix = f"{category}/" if category else ""

    # Schema-v3 kernel tasks declare task.yaml; schema-v2 serving tasks keep
    # config.yaml. A directory must not declare both.
    for name in ("task.yaml", "config.yaml"):
        for config_path in root.glob(f"{prefix}**/{name}"):
            task_name = config_path.parent.relative_to(root).as_posix()
            if task_name in task_config_dict:
                raise ValueError(f"Task {task_name} declares both task.yaml and config.yaml")
            task_config_dict[task_name] = str(config_path)

    return dict(sorted(task_config_dict.items()))

