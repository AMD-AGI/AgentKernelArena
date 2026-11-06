# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
# utils/registry.py

from pathlib import Path
from typing import Dict, Optional


def get_task_config(tasks_root: str = "tasks", category: Optional[str] = None) -> Dict[str, str]:
    """Find every task under ``tasks_root``.

    A schema-v3 kernel task declares ``task.yaml``; a schema-v2 e2e serving task
    keeps ``config.yaml``. The task name is the directory's path relative to
    ``tasks_root``, for example ``tasks/triton/normalization/rms_norm/bf16_h8192/task.yaml``
    -> ``triton/normalization/rms_norm/bf16_h8192``.

    Args:
        tasks_root: Root directory for tasks (default: "tasks")
        category: Optional path prefix to filter tasks, matched as a glob over whole
            path components (e.g. "triton", "triton/normalization", "*/normalization"
            for one category under every baseline language, or "e2e")

    Returns:
        dict: {task_name: path of its task.yaml or config.yaml}
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


def task_category(task_name: str) -> str:
    """The operator category in a task name, "e2e" for a serving task, else the first segment.

    Kernel task names are <baseline_language>/<category>/<operator>/<scenario>,
    so the category is the second segment; returns "" when there is no slash.
    """
    from src.task_contract import BASELINE_LANGUAGES

    if not isinstance(task_name, str) or "/" not in task_name:
        return ""
    parts = task_name.split("/")
    if len(parts) >= 4 and parts[0] in BASELINE_LANGUAGES:
        return parts[1]
    return parts[0]
