# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import yaml
import logging
from pathlib import Path
from src.task_spec import load_task_spec
from src.task_prompt import build_task_prompt


def _load_cheatsheet(language: str, target_gpu_model: str, project_root: Path,
                     logger: logging.Logger) -> tuple[str, str | None]:
    """
    Load the combined cheatsheet prompt and resolve the target gfx arch string.

    The cheatsheet config (default_cheatsheet.yaml) has two independent sections:
      - architecture: maps GPU model name → hardware spec document(s) + gfx_arch string
      - knowledge:    maps candidate language → language best-practice guide (GPU-agnostic)

    An architecture entry may override a language guide (``knowledge_override``).

    Returns:
        (cheatsheet_text, gfx_arch)  where gfx_arch may be None if not found.
    """
    gfx_arch = None
    try:
        cheatsheet_config_path = project_root / "src/prompts/cheatsheet/default_cheatsheet.yaml"
        cheatsheet_config = yaml.safe_load(cheatsheet_config_path.read_text()) or {}

        parts: list[str] = []

        # --- Architecture section ---
        arch_map = cheatsheet_config.get('architecture', {})
        gpu_key = str(target_gpu_model)
        arch_entry = (
            arch_map.get(gpu_key)
            or arch_map.get(gpu_key.upper())
            or arch_map.get(gpu_key.lower())
        )
        if arch_entry:
            gfx_arch = arch_entry.get('gfx_arch')
            arch_files = arch_entry.get('files')
            if arch_files is None:
                single_file = arch_entry.get('file')
                arch_files = [single_file] if single_file else []
            elif isinstance(arch_files, str):
                arch_files = [arch_files]

            for arch_file in arch_files:
                arch_path = project_root / arch_file
                parts.append(arch_path.read_text())
                logger.info(f"Loaded architecture context for '{target_gpu_model}': {arch_path}")
            if not arch_files:
                logger.warning(
                    f"Architecture entry for '{target_gpu_model}' has no 'file' or 'files' key"
                )
        else:
            logger.warning(f"No architecture entry for GPU '{target_gpu_model}' in default_cheatsheet.yaml")

        # --- Knowledge section ---
        target_language = language.lower()
        knowledge_override = arch_entry.get('knowledge_override', {}) if arch_entry else {}
        knowledge_map = cheatsheet_config.get('knowledge', {})
        knowledge_file = knowledge_override.get(target_language) or knowledge_map.get(target_language)
        if knowledge_file:
            knowledge_path = project_root / knowledge_file
            parts.append(knowledge_path.read_text())
            logger.info(f"Loaded knowledge cheatsheet for '{target_language}': {knowledge_path}")
        else:
            logger.warning(f"No knowledge cheatsheet for language '{target_language}' in default_cheatsheet.yaml")

        cheatsheet_text = "\n\n---\n\n".join(parts) if parts else ""

    except Exception as e:
        logger.warning(f"Failed to load cheatsheet: {e}, using empty cheatsheet")
        cheatsheet_text = ""
        gfx_arch = None

    return cheatsheet_text, gfx_arch


def prompt_builder(task_config_dir: str, workspace_directory: Path, eval_config: dict, logger: logging.Logger) -> str:
    """
    Build the initial prompt for the agent based on task configuration.

    Args:
        task_config_dir: Path to the task's task.yaml (schema v3) or config.yaml (schema v2)
        workspace_directory: Path to the materialized workspace for the agent
        eval_config: Evaluator-level config (contains target_gpu_model, etc.)
        logger: Logger instance

    Returns:
        str: The complete prompt for the agent, rendered from the task
        declaration plus the architecture and candidate-language cheatsheets.
    """
    task_config_path = Path(task_config_dir)
    with open(task_config_path, 'r') as f:
        task_config = yaml.safe_load(f) or {}

    target_gpu_model = eval_config.get('target_gpu_model')
    if not target_gpu_model:
        raise ValueError(
            "target_gpu_model is required in config.yaml. "
            "Set it to your GPU model (e.g. MI300, MI355X, RDNA4)."
        )
    logger.info(f"Building prompt from config: {task_config_path}")

    schema_version = task_config.get('schema_version')
    if schema_version not in (2, 3):
        raise ValueError(f"Unsupported task schema version: {schema_version!r}")
    project_root = Path(__file__).resolve().parent.parent
    stable_id = eval_config.get('_task_id')
    if stable_id is None:
        stable_id = task_config_path.resolve().parent.relative_to(project_root / 'tasks').as_posix()
    if schema_version == 3:
        # The workspace holds the execution record resolved for this run's
        # target language; the task package itself names no language.
        spec = load_task_spec(Path(workspace_directory) / 'config.yaml', task_id=stable_id)
    else:
        spec = load_task_spec(task_config_path, task_id=stable_id)
    context, gfx_arch = _load_cheatsheet(spec.candidate.language, target_gpu_model, project_root, logger)
    return build_task_prompt(spec, Path(workspace_directory), target_gpu=target_gpu_model,
                             hardware_context=context, target_arch=gfx_arch)
