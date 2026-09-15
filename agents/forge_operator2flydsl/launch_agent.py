"""Deprecated public name for the unified Forge v2 adapter."""
from agents import register_agent
from agents.forge.launch_agent import launch_agent as _launch


@register_agent("forge_operator2flydsl")
def launch_agent(eval_config: dict, task_config_dir: str, workspace: str) -> str:
    return _launch(eval_config, task_config_dir, workspace)
