# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import yaml
import logging
from pathlib import Path
from src.prompts import task_type


def source_code(config: dict) -> str:
    """Generate a comprehensive task definition prompt for HIP kernel optimization."""

    # Format lists properly
    source_files = "\n    - ".join(config['source_file_path']) if isinstance(config['source_file_path'], list) else config['source_file_path']
    target_kernels = "\n    - ".join(config['target_kernel_functions']) if isinstance(config['target_kernel_functions'], list) else config['target_kernel_functions']

    return f"""
### Source Code
**File(s) to optimize:**
    - {source_files}

**Target kernel function(s):**
    - {target_kernels}
"""


def instructions(config: dict) -> str:

    # Handle multiple compile/test commands
    compile_cmds = "\n    ".join(config['compile_command']) if isinstance(config['compile_command'], list) else config['compile_command']
    correctness_cmds = "\n    ".join(config['correctness_command']) if isinstance(config['correctness_command'], list) else config['correctness_command']
    performance_cmds = "\n    ".join(config.get('performance_command', ['Not specified'])) if isinstance(config.get('performance_command'), list) else config.get('performance_command', 'Not specified')

    return f"""
### Instructions

1. **Compilation Phase**
    Execute the following command(s) to build the optimized code:
    {compile_cmds}
    - Compilation must succeed without errors or warnings.

2. **Correctness Validation Phase** (MANDATORY)
    Execute the following command(s) to verify correctness:
    {correctness_cmds}
    - The output must indicate "Validation passed" or equivalent success message.
    - Any validation errors are unacceptable - the optimization must produce identical results to the original.
    - Compare kernel execution time before and after optimization.

3. **Performance Analysis Phase** (OPTIONAL)
    Execute the following command(s) to measure performance:
    {performance_cmds}
    - Profiling could help you to identify the bottlenecks and optimize the kernel.
"""


def task_result_format(task_result_template: str) -> str:
    """
    Generate the output format section.
    
    Note: Agents should NOT write task_result.yaml anymore - the framework handles this.
    This function now just provides information about what will be evaluated.

    Returns:
        str: Formatted output requirements section
    """
    return f"""
### Completion

**IMPORTANT**: After you complete the kernel optimization:

1. **Save your optimized kernel code** in the workspace directory.
2. **DO NOT write task_result.yaml** - the framework will automatically:
   - Check compilation
   - Validate correctness
   - Measure performance
   - Generate task_result.yaml with standardized metrics

Your job is complete once you have optimized the kernel code. The framework will handle all evaluation and scoring automatically.

"""


def _load_cheatsheet(task_type_name: str, target_gpu_model: str, project_root: Path,
                     task_config: dict, logger: logging.Logger) -> tuple[str, str | None]:
    """
    Load the combined cheatsheet prompt and resolve the target gfx arch string.

    The cheatsheet config (default_cheatsheet.yaml) has two independent sections:
      - architecture: maps GPU model name → hardware spec document + gfx_arch string
      - knowledge:    maps target language → language best-practice guide (GPU-agnostic)

    Returns:
        (cheatsheet_text, gfx_arch)  where gfx_arch may be None if not found.
    """
    # Task config can override the whole cheatsheet inline.
    override = task_config.get('prompt', {}).get('cheatsheet')
    if override is not None:
        return override, None

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
            arch_file = arch_entry.get('file')
            if arch_file:
                arch_path = project_root / arch_file
                parts.append(arch_path.read_text())
                logger.info(f"Loaded architecture context for '{target_gpu_model}': {arch_path}")
            else:
                logger.warning(f"Architecture entry for '{target_gpu_model}' has no 'file' key")
        else:
            logger.warning(f"No architecture entry for GPU '{target_gpu_model}' in default_cheatsheet.yaml")

        # --- Knowledge section ---
        target_language = (
            task_type_name.split('2')[-1] if '2' in task_type_name else task_type_name
        ).lower()
        knowledge_map = cheatsheet_config.get('knowledge', {})
        knowledge_file = knowledge_map.get(target_language)
        if knowledge_file:
            knowledge_path = project_root / knowledge_file
            parts.append(knowledge_path.read_text())
            logger.info(f"Loaded knowledge cheatsheet for '{target_language}': {knowledge_path}")
        else:
            logger.warning(f"No knowledge cheatsheet for language '{target_language}' in default_cheatsheet.yaml")

        cheatsheet_text = "\n\n---\n\n".join(parts) if parts else ""

    except ValueError:
        raise
    except Exception as e:
        logger.warning(f"Failed to load cheatsheet: {e}, using empty cheatsheet")
        cheatsheet_text = ""
        gfx_arch = None

    return cheatsheet_text, gfx_arch


def _gpu_arch_precheck_prompt(target_gpu_model: str, gfx_arch: str | None) -> str:
    """
    Generate a pre-task directive instructing the agent to detect and fix any
    hardcoded GPU architecture strings in the workspace build files before
    running compile / test / benchmark commands.

    Returns an empty string when gfx_arch is unknown (nothing to check against).
    """
    if not gfx_arch:
        return ""

    return f"""
### Pre-Task Setup: GPU Architecture Consistency Check

**Target GPU:** `{target_gpu_model}` — architecture token: `{gfx_arch}`

**Before running any build, test, or benchmark command**, perform the following check:

1. Scan all build-related files in the workspace for hardcoded GPU architecture strings.
   Focus especially on:
   - `Makefile` — variables such as `AMDGPU_TARGETS`, `ROCM_ARCH`,
     `HIPCC_COMPILE_FLAGS_APPEND`, `PYTORCH_ROCM_ARCH`, and flags like
     `--offload-arch=<arch>` or `-DAMDGPU_TARGETS=<arch>`
   - `CMakeLists.txt` / `*.cmake` — `AMDGPU_TARGETS`, `GPU_TARGETS`, `--offload-arch`
   - Shell scripts and Python test scripts — cmake invocations with `-DAMDGPU_TARGETS=`

2. If **any** file contains a hardcoded GPU architecture that **differs** from
   `{gfx_arch}`, update that file to use `{gfx_arch}` before proceeding with
   any other step.

3. Only after confirming that all build files target `{gfx_arch}` (or were already
   correct) should you proceed with the task.
"""


def prompt_builder(task_config_dir: str, workspace_directory: Path, eval_config: dict, logger: logging.Logger) -> str:
    """
    Build the initial prompt for the agent based on task configuration.

    Args:
        task_config_dir: Path to the task's config.yaml
        workspace_directory: Path to the duplicated workspace for the agent
        eval_config: Evaluator-level config (contains target_gpu_model, etc.)
        logger: Logger instance

    Returns:
        str: The complete prompt for the agent

    Prompt section order:
        1. Task Type
        2. Source Code
        3. GPU Arch Pre-check  ← new: fix mismatched arch before first build
        4. Instructions
        5. Output Format
        6. Cheatsheet  (architecture context + language knowledge, combined)
        7. Workspace Directory
    """
    # Load task configuration
    task_config_path = Path(task_config_dir)
    with open(task_config_path, 'r') as f:
        task_config = yaml.safe_load(f)

    task_type_name = task_config.get('task_type')
    target_gpu_model = eval_config.get('target_gpu_model', 'MI300')
    logger.info(f"Building prompt from config: {task_config_path}")

    # Build prompt sections
    prompt_sections = []

    # 1. Task Type Section
    if not task_type_name:
        raise ValueError("task_type is missing in task config")
    if task_type_name == 'hip2hip':
        task_type_prompt = task_type.hip2hip_task_type()
    elif task_type_name == 'torch2hip':
        task_type_prompt = task_type.torch2hip_task_type()
    elif task_type_name == 'triton2triton':
        task_type_prompt = task_type.triton2triton_task_type()
    elif task_type_name == 'instruction2triton':
        task_type_prompt = task_type.instruction2triton_task_type()
    else:
        raise ValueError(f"Unknown task type: {task_type_name}")

    prompt_sections.append(task_type_prompt)

    # 2. Source Code Section
    source_code_prompt = task_config.get('prompt', {}).get('source_code')
    source_file_path = task_config.get('source_file_path', [])
    if source_code_prompt is None:
        if any(x is not None for x in source_file_path):
            source_code_prompt = source_code(task_config)
        else:
            source_code_prompt = ""

    prompt_sections.append(source_code_prompt)

    # 3. Cheatsheet: architecture context + language knowledge
    project_root = Path(__file__).resolve().parent.parent
    cheatsheet_prompt, gfx_arch = _load_cheatsheet(
        task_type_name, target_gpu_model, project_root, task_config, logger
    )

    # 4. GPU Arch Pre-check (inserted before instructions so the agent sees it first)
    precheck_prompt = _gpu_arch_precheck_prompt(target_gpu_model, gfx_arch)
    if precheck_prompt:
        prompt_sections.append(precheck_prompt)

    # 5. Instructions Section
    instructions_prompt = task_config.get('prompt', {}).get('instructions')
    if instructions_prompt is None:
        instructions_prompt = instructions(task_config)

    prompt_sections.append(instructions_prompt)

    # 6. Output Format Section
    task_result_template = task_config.get('task_result_template', '')
    task_result_prompt = task_config.get('prompt', {}).get('output_format')
    if task_result_prompt is None:
        task_result_prompt = task_result_format(task_result_template)

    prompt_sections.append(task_result_prompt)

    # 7. Cheatsheet Section (architecture + knowledge combined)
    prompt_sections.append(cheatsheet_prompt)

    # 8. Workspace Directory Information
    workspace_info = f"""
### Workspace Directory
Your working directory is: `{workspace_directory}`

This duplicated workspace contains:
- All source files that need to be optimized
- Build system and compilation tools
- Test and validation scripts
- Profiling tools

You can directly modify the kernel code in this workspace. All changes should be made within this directory.

"""
    prompt_sections.append(workspace_info)

    # Combine all sections
    final_prompt = "\n\n".join(prompt_sections)

    logger.info(f"Prompt built successfully, total length: {len(final_prompt)} characters")

    return final_prompt
