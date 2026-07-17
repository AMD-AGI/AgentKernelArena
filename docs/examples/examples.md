---
myst:
    html_meta:
        "description": "Step-by-step examples for running AgentKernelArena experiments, A/B testing agent capabilities, validating tasks, and resuming interrupted runs."
        "keywords": "AgentKernelArena, examples, experiment, A/B test, task validation, GPU kernel, ROCm, HIP, Triton"
---

# AgentKernelArena examples

These walkthroughs assume you have completed [installation](../install/install.md)
and can run the Docker runner (`make docker-smoke` passes). The agent selected
by each example must also be installed and authenticated on the host; substitute
another installed agent when appropriate. Serial examples use `make docker-run`;
multi-GPU examples use `make docker-parallel-run`.

## Example 1: Run one agent configuration on one task

Run the single-task Claude Code quickstart for the physical GPU.

1. Select the MI300/MI300X example (use
   `example_configs/quickstart_claude_mi355x.yaml` on MI355X):

    ```bash
    CONFIG_PATH=example_configs/quickstart_claude_mi300.yaml
    ```

2. Run:

    ```bash
    make docker-check-agents CONFIG="$CONFIG_PATH"
    make docker-run CONFIG="$CONFIG_PATH"
    ```

3. Inspect the result:

    ```text
    workspace_MI300_claude_code/
    └── run_<timestamp>/
        └── hip2hip_gpumode_GELU_<timestamp>/
            └── task_result.yaml
    ```

A successful `task_result.yaml` looks like:

```yaml
task_name: hip2hip/gpumode/GELU
pass_compilation: true
pass_correctness: true
base_execution_time: 1.82
best_optimized_execution_time: 1.15
speedup_ratio: 1.58
score: 278.0
```

## Example 2: Run an experiment on a whole task category

Run Claude Code across all vLLM Triton tasks. Save this configuration as
`config_triton.yaml`:

```yaml
agent:
  template: claude_code

tasks:
  - triton2triton/vllm

target_gpu_model: MI300
log_directory: logs
workspace_directory_prefix: workspace
```

```bash
make docker-run CONFIG=config_triton.yaml
```

Each task gets its own workspace under the same `run_<timestamp>/` directory, and
post-processing aggregates them into a run report.

## Example 3: A/B test an agent capability

Measure whether a new Model Context Protocol (MCP) server, skill, or prompt
change helps. Run the same task set twice with distinct run suffixes.

```bash
# Baseline (capability disabled in the agent configuration)
make docker-run CONFIG=config_triton.yaml RUN_ARGS="--run-suffix baseline"

# Treatment (capability enabled)
make docker-run CONFIG=config_triton.yaml RUN_ARGS="--run-suffix with_capability"
```

Both runs land in `workspace_MI300_<agent>/` with distinct run names. Build the
dashboard and compare them side-by-side:

```bash
python3 -m src.visualization build --include-workspace-runs
python3 -m src.visualization serve --host 127.0.0.1 --port 8080
```

Open <http://127.0.0.1:8080> to compare scores. See
[Visualize and compare runs](../how-to/visualization.md) for details.

## Example 4: Validate a new task

Before merging a new task, run the task_validator agent against it. Save this
configuration as `config_validator.yaml`:

```yaml
agent:
  template: task_validator

tasks:
  - hip2hip/others/my_new_kernel

target_gpu_model: MI300
log_directory: logs
workspace_directory_prefix: workspace
```

```bash
make docker-run CONFIG=config_validator.yaml
```

Open the generated `validation_report.yaml` in the task workspace. The task must
reach **PASS** (or **WARN** with justification) before merging. See
[Validate tasks](../how-to/task-validator.md) for more information.

## Example 5: Run eight GPU workers in parallel

On an 8-GPU server, start one Docker worker per GPU and let the workers share the
same task queue. Save this configuration as
`config_parallel_claude_mi355x.yaml`:

```yaml
agent:
  template: claude_code

tasks:
  - hip2hip/gpumode

target_gpu_model: MI355X
log_directory: logs
workspace_directory_prefix: workspace
```

```bash
make docker-parallel-run \
  CONFIG=config_parallel_claude_mi355x.yaml \
  GPU_IDS=0,1,2,3,4,5,6,7 \
  RUN_ARGS="--run-suffix claude_parallel8"
```

The run directory contains normal per-task workspaces plus a scheduler queue:

```text
workspace_MI355X_claude_code/
└── run_<timestamp>_claude_parallel8/
    ├── .parallel/
    │   ├── pending/
    │   ├── running/
    │   ├── done/
    │   └── failed/
    ├── hip2hip_gpumode_GELU_<timestamp>/
    │   └── task_result.yaml
    └── reports/
        └── overall_report.txt
```

Each worker sees one logical GPU inside its container (`HIP_VISIBLE_DEVICES=0`,
`CUDA_VISIBLE_DEVICES=0`) while `ROCR_VISIBLE_DEVICES` selects the host GPU.
Post-processing runs once after all worker containers finish.

## Example 6: Validate tasks in parallel

The same queue works for `task_validator`. This is useful before merging a large
batch of tasks. Save this configuration as
`config_parallel_validator_mi355x.yaml`:

```yaml
agent:
  template: task_validator

tasks:
  - hip2hip/gpumode

target_gpu_model: MI355X
log_directory: logs
workspace_directory_prefix: workspace
```

```bash
make docker-parallel-run \
  CONFIG=config_parallel_validator_mi355x.yaml \
  GPU_IDS=0,1,2,3,4,5,6,7 \
  RUN_ARGS="--run-suffix validator_parallel8"
```

Each task workspace writes `validation_report.yaml`, and the run directory gets
one final `validation_summary.yaml` after all workers complete.

## Example 7: Resume an interrupted run

If a long run is interrupted, resume it without repeating completed tasks:

```bash
# Resume the most recent run
make docker-run CONFIG=example_configs/quickstart_claude_mi300.yaml RUN_ARGS="--resume-latest"

# Or resume a specific run directory
make docker-run CONFIG=example_configs/quickstart_claude_mi300.yaml RUN_ARGS="--resume-run run_20260617_101500"
```

For a parallel run, use the same resume arguments with `docker-parallel-run`:

```bash
make docker-parallel-run CONFIG=config_parallel_claude_mi355x.yaml GPU_IDS=0,1,2,3 RUN_ARGS="--resume-latest"
make docker-parallel-run CONFIG=config_parallel_claude_mi355x.yaml GPU_IDS=0,1,2,3 RUN_ARGS="--resume-run run_20260617_101500_parallel8"
```
