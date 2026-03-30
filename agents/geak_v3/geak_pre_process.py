# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""
GEAK Benchmark Pre-Processing Module.

This module handles preprocessing for GEAK benchmark tasks:
1. Building simplified prompts from task config
2. Copying python_bindings to workspace
3. Integrating agent config into prompts
"""
import shutil
import logging
from pathlib import Path
from typing import Any
import yaml


def simple_prompt_builder(task_config_dir: str, workspace: str, logger: logging.Logger) -> str:
    """
    Build a simple prompt for geak_v3 agent.
    Only includes essential information from task config.

    Args:
        task_config_dir: Path to the task's config.yaml
        workspace: Workspace directory path
        logger: Logger instance

    Returns:
        str: The simplified prompt
    """
    task_config_path = Path(task_config_dir)
    with open(task_config_path, 'r') as f:
        task_config = yaml.safe_load(f)

    prompt_sections = []

    # 1. Task info from config
    source_files = task_config.get('source_file_path', [])
    target_kernels = task_config.get('target_kernel_functions', [])
    compile_cmd = task_config.get('compile_command', [])
    correctness_cmd = task_config.get('correctness_command', [])
    performance_cmd = task_config.get('performance_command', [])

    # Format as list strings
    def format_list(items):
        if isinstance(items, list):
            return '\n'.join(f'  - {item}' for item in items)
        return f'  - {items}'

    # Normalize source file paths to absolute paths in workspace context.
    def absolutize_source_paths(items, workspace_dir: str):
        if items is None:
            return []
        raw_items = items if isinstance(items, list) else [items]
        workspace_path = Path(workspace_dir)
        abs_items = []
        for item in raw_items:
            path_str = str(item).strip()
            if not path_str:
                continue
            p = Path(path_str)
            abs_items.append(str(p if p.is_absolute() else (workspace_path / p)))
        return abs_items

    source_files = absolutize_source_paths(source_files, workspace)

    # Build test command: compile_command && correctness_command && performance_command (dedup identical cmds)
    def build_test_command(compile_cmds, correctness_cmds, perf_cmds):
        def normalize(cmds):
            if cmds is None:
                return []
            if isinstance(cmds, list):
                raw = cmds
            else:
                raw = [cmds]
            out = []
            for c in raw:
                s = str(c).strip()
                if s:
                    out.append(s)
            return out

        ordered = []
        seen = set()
        for cmd in normalize(compile_cmds) + normalize(correctness_cmds) + normalize(perf_cmds):
            if cmd in seen:
                continue
            seen.add(cmd)
            ordered.append(cmd)
        return " && ".join(ordered)

    test_command = build_test_command(compile_cmd, correctness_cmd, performance_cmd)

    task_info = f"""## Task Info

**Kernel_url:**
{format_list(source_files)}

**Target kernel functions:**
{format_list(target_kernels)}

**Test command:**
  - `{test_command}`
"""
    prompt_sections.append(task_info)

    # 2. Custom instructions from task config (if provided)
    instructions = task_config.get('prompt', {}).get('instructions')
    if instructions:
        prompt_sections.append(f"## Instructions\n\n{instructions}")
    else:
        prompt_sections.append("Optimize the kernel in the workspace directory.")
    
    # 3. Workspace directory info
    workspace_info = f"""
### Workspace Directory
Your working directory is: `{workspace}`
"""
    prompt_sections.append(workspace_info)

    final_prompt = "\n\n".join(prompt_sections)
    logger.info(f"Simple prompt built, length: {len(final_prompt)} characters")

    return final_prompt


def integrate_agent_config(prompt: str, agent_config: dict[str, Any]) -> str:
    """
    Integrate agent config into prompt.
    
    Args:
        prompt: The base prompt string
        agent_config: Agent configuration dictionary
        
    Returns:
        str: Updated prompt with agent config integrated
    """
    max_iters = agent_config.get("max_iterations")
    if max_iters is not None:
        prompt = prompt.rstrip() + f"\n\nFor this optimization, you must iterate up to {max_iters} versions."
    python_path = agent_config.get("python_path")
    if python_path:
        prompt = prompt.rstrip() + f"\n\nUse this Python interpreter: `{python_path}`."
    return prompt


def copy_python_bindings(task_config_dir: str, workspace: str, logger: logging.Logger) -> None:
    """
    Copy python_bindings directory from task folder to workspace if it exists.
    
    Args:
        task_config_dir: Path to the task's config.yaml
        workspace: Workspace directory path
        logger: Logger instance
    """
    task_config_path = Path(task_config_dir)
    python_bindings_src = task_config_path.parent / "python_bindings"

    if python_bindings_src.exists() and python_bindings_src.is_dir():
        python_bindings_dst = Path(workspace) / "python_bindings"
        python_bindings_dst.mkdir(parents=True, exist_ok=True)

        for item in python_bindings_src.iterdir():
            dst = python_bindings_dst / item.name
            if item.is_dir():
                shutil.copytree(item, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dst)

        logger.info(f"Copied python_bindings from {python_bindings_src} to {python_bindings_dst}")
