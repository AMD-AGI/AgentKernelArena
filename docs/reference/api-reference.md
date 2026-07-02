---
myst:
    html_meta:
        "description": "Complete reference for AgentKernelArena configuration: run config.yaml schema, task config fields, CLI flags, scoring formula, and the agent registry."
        "keywords": "AgentKernelArena, API reference, config.yaml, CLI flags, scoring, agent registry, ROCm, GPU kernel"
---

# AgentKernelArena configuration and API reference

This topic documents the run configuration (`config.yaml`), the per-task
configuration, the command-line flags, the scoring formula, and the agent
registry.

## Run configuration (`config.yaml`)

The root `config.yaml` defines a single evaluation run.

| Field | Type | Description |
| --- | --- | --- |
| `agent.template` | string | Agent to run. One of the [supported agents](../how-to/agents.md#supported-agents). |
| `tasks` | list of strings | Task selectors relative to `tasks/`. Use `all` for every task, a category prefix for a group, or a full path for a single task. |
| `target_gpu_model` | string | Target GPU model, for example `MI300` or `MI355X`. Used to select the Docker image architecture, set `PYTORCH_ROCM_ARCH`, and name the workspace. |
| `log_directory` | string | Directory for run logs. |
| `workspace_directory_prefix` | string | Prefix for the workspace directory. The full name is `<prefix>_<gpu>_<agent>`. |

Example:

```yaml
agent:
  template: cursor

tasks:
  - hip2hip/gpumode/GELU
  - triton2triton/vllm/triton_rms_norm

target_gpu_model: MI300
log_directory: logs
workspace_directory_prefix: workspace
```

## Command-line flags

The in-container `main.py` entrypoint accepts these flags:

| Flag | Description |
| --- | --- |
| `--config_name <file>` | Config file to load (default `config.yaml`). Use separate files to keep multiple task sets in one folder |
| `--run-suffix <suffix>` | Suffix appended to the run directory name (letters, numbers, `.`, `_`, `-` only). Useful for labeling A/B runs |
| `--resume-run <run_dir>` | Resume a specific run directory, skipping completed tasks |
| `--resume-latest` | Resume the most recent run in the workspace |

These flags are passed to the in-container entrypoint via `make docker-run`
(`CONFIG=` sets `--config_name`; `RUN_ARGS=` forwards the rest):

```bash
make docker-run CONFIG=config_triton.yaml RUN_ARGS="--run-suffix with_mcp"
```

## Task configuration

Each task is defined by a `config.yaml` in its directory. Command fields are
*lists*.

For isolated-kernel tasks (`hip2hip`, `cuda2hip`, `triton2triton`,
`instruction2triton`, `torch2hip`, and `flydsl2flydsl`):

| Field | Required | Description |
| --- | --- | --- |
| `source_file_path` | Yes | Source files containing the kernel, relative to the task root |
| `target_kernel_functions` | Yes | Kernel function names that must be defined in the source |
| `compile_command` | Yes | Command(s) to compile or build-check |
| `correctness_command` | Yes | Command(s) to validate correctness |
| `task_type` | Yes | One of `hip2hip`, `cuda2hip`, `triton2triton`, `instruction2triton`, `torch2hip`, or `flydsl2flydsl` |
| `performance_command` | No | Command(s) to measure performance |
| `task_result_template` | No | Override the result template (`null` = default) |
| `platform_support` | No | Optional run-gating metadata; see below |
| `prompt.source_code` | No | Override the prompt's source-code section |
| `prompt.instructions` | No | Custom prompt instructions |
| `prompt.cheatsheet` | No | Reference/cheatsheet content for the prompt |

For repository-level tasks (`task_type: repository`):

| Field | Required | Description |
| --- | --- | --- |
| `repo_url` | Yes | Upstream repository to clone for the task |
| `task_type` | Yes | Must be `repository` |
| `repository_language` | Yes | Primary optimization stack, for example `hip` or `triton` |
| `compile_command` | Yes | Command(s) to compile or build-check |
| `correctness_command` | Yes | Command(s) to validate correctness |
| `performance_command` | No | Command(s) to measure performance |
| `post_clone_install` | No | Setup command(s) to run after cloning the upstream repository |
| `post_clone_install_mode` | No | Controls when `post_clone_install` runs, for example `every_setup` |
| `source_file_path` | No | Optional target source-file hints, relative to the cloned repository root |
| `target_kernel_functions` | No | Optional target function or kernel-symbol hints |
| `platform_support` | No | Optional run-gating metadata; see below |
| `prompt.instructions` | No | Custom prompt instructions |
| `prompt.cheatsheet` | No | Reference/cheatsheet content for the prompt |

Optional platform support metadata:

```yaml
platform_support:
  required_arch: gfx942   # optional; compared with target_gpu_model's gfx arch
  status: active          # optional; active or skip
  skip_reason: null       # recommended when status is skip
```

If `platform_support` is omitted, the task is scheduled normally. If
`status: skip`, or if `required_arch` does not match the current run's resolved
GPU architecture, the task is skipped before workspace creation and agent
launch. Historical task-specific fields such as `runnable_on_gfx942` are
documentation only and do not affect scheduling.

See [Add a task](../how-to/add-task.md) for layout and authoring rules.

## Result schema (`task_result.yaml`)

Each task produces a `task_result.yaml` in its workspace:

| Field | Description |
| --- | --- |
| `task_name` | `<task_type>/<task_name>` |
| `pass_compilation` | Whether the optimized kernel compiled |
| `compilation_error_message` | Error text if compilation failed, else `null` |
| `pass_correctness` | Whether correctness passed |
| `correctness_error_message` | Error text if correctness failed, else `null` |
| `base_execution_time` | Baseline runtime in ms |
| `best_optimized_execution_time` | Best optimized runtime in ms |
| `speedup_ratio` | Speedup over baseline |
| `baseline_benchmark_methods` | Timing methods observed while measuring the baseline |
| `optimized_benchmark_methods` | Timing methods observed while measuring the optimized kernel |
| `benchmark_method_consistent` | Whether baseline and optimized timing methods matched |
| `valid_baseline_cases` | Number of baseline test cases with usable timing results |
| `valid_optimized_cases` | Number of optimized test cases with usable timing results |
| `speedup_calculation_error_message` | Error text if speedup could not be calculated, else `null` |
| `optimization_summary` | Free-text summary from the agent |
| `score` | Computed score (see below) |

## Scoring

The score is the sum of three components:

| Component | Points | Condition |
| --- | --- | --- |
| Compilation | `20` | The kernel compiles successfully |
| Correctness | `100` | The kernel passes the correctness check |
| Speedup | `speedup_ratio × 100` | Added only when compilation *and* correctness pass |

The rules, expressed as the framework applies them:

- Compilation fails → score `0`.
- Compilation passes, correctness fails → score `20`.
- Both pass → `120 + speedup_ratio × 100`.

**Example**: A kernel that compiles (`20`), is correct (`100`), and achieves a
`1.58×` speedup scores `20 + 100 + 158 = 278`.

The speedup used for scoring prefers the explicit `speedup_ratio` written by the
evaluator (which weights each test case equally for multi-testcase tasks) and
falls back to `base_execution_time / best_optimized_execution_time` when an
explicit ratio is not present.

This is the default scoring scheme; you can define your own in `src/score.py`.

## Agent registry

Agents register themselves into a shared registry with the `register_agent`
decorator, and the framework loads only the selected agent:

```python
from agents import register_agent

@register_agent("your_agent")
def launch_agent(eval_config, task_config_dir, workspace):
    ...
    return result
```

The selectable agent names are defined by the `AgentType` enum in
`src/module_registration.py`. See [Configure agents and models](../how-to/agents.md)
for the integration steps.
