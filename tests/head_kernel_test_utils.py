"""Locate relocated head-kernel test inputs using stable operation metadata."""
from pathlib import Path

import yaml


SUITE = Path(__file__).resolve().parents[1] / "tasks/head_kernels"


def task_directories():
    tasks = {}
    for config in sorted(SUITE.rglob("config.yaml")):
        operation_id = yaml.safe_load(config.read_text())["headkernel"]["operation_id"]
        if operation_id in tasks:
            raise AssertionError(f"duplicate head-kernel operation ID: {operation_id}")
        tasks[operation_id] = config.parent
    return tasks


def task_directory(operation_id):
    return task_directories()[operation_id]
