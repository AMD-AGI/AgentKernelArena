# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Forge registration; the adapter consumes the framework's schema-v3 task context."""

from __future__ import annotations

from typing import Any

from agents import register_agent


@register_agent("forge")
def launch_agent(eval_config: dict[str, Any], task_config_dir: str, workspace: str) -> str:
    """Run the Forge adapter on the framework-validated task in ``workspace``."""
    from agents.forge.adapter import launch
    return launch(eval_config, task_config_dir, workspace)
