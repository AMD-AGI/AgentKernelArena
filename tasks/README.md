# Tasks Folder Guide

This document explains the `tasks/` folder layout and how to add new tasks to AgentKernelArena.

## Folder Structure

`tasks/` is organized by task type, then task family, then task name.

Typical pattern:

`tasks/<task_type>/<task_name>/`

Examples:

- `tasks/hip2hip/rmsnorm/`
- `tasks/cuda2hip/awq_gemm/`
- `tasks/triton2triton/triton_decode_attention/`


Common task types currently used:

- `hip2hip`
- `cuda2hip`
- `triton2triton`
- `torch2hip`
- `instruction2triton`

If a required task type folder does not exist, create it.

## Expected Files in a Task

At minimum, each task should contain:

- `config.yaml` (required)

Recommended for self-contained tasks:

- `source/` (kernel or reference source files)
- `scripts/` (compile/correctness/performance runners)
- `README.md` (task-specific notes and assumptions)

## Required `config.yaml` Fields

Every task config should include:

- `source_file_path`
- `target_kernel_functions`
- `compile_command`
- `correctness_command`
- `performance_command`
- `task_type`
- `task_result_template`
- `prompt` (`source_code`, `instructions`, `cheatsheet`)

Keep commands runnable from task directory context and deterministic whenever possible.

## Correctness Guidance (Important)

Correctness checks should be numeric and reproducible:

1. Generate random inputs with fixed seed.
2. Compute CPU reference output (Python or C++).
3. Compute candidate output.
4. Compare with tolerance (`atol`/`rtol`, e.g. `1e-4`).

Avoid “symbol exists” checks as the final correctness criterion.

## Performance Guidance

Performance should be measured separately from correctness:

- warmup iterations
- timed iterations
- average/median timing
- fixed shape suites (including holdout shapes if applicable)


## How to Add a New Task

1. Choose task type (`hip2hip`, `cuda2hip`, `triton2triton`, etc.).
2. Create folder: `tasks/<task_type>/<task_name>/`.
3. Add `config.yaml` with required fields.
4. Add source files and runner scripts.
5. Ensure `compile_command`, `correctness_command`, and `performance_command` all run successfully.
6. Add task-level `README.md` describing:
   - what kernel is evaluated
   - input/output contract
   - correctness tolerance
   - known limitations
7. Run validation locally before committing.

## Quality Checklist Before Commit

- `config.yaml` fields are complete and accurate.
- Commands are executable in a clean environment.
- Correctness uses CPU reference + numeric threshold.
- Performance report is reproducible.
- No hidden external runtime dependencies unless explicitly documented.

