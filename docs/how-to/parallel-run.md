---
myst:
    html_meta:
        "description": "Run AgentKernelArena experiments across multiple GPUs with one Docker worker container per GPU, dynamic task claiming, isolated caches, and shared post-processing."
        "keywords": "AgentKernelArena, multi-GPU, parallel run, Docker worker, GPU isolation, task_validator, ROCm"
---

# Run tasks in parallel across multiple GPUs in AgentKernelArena

Use `make docker-parallel-run` when one machine has multiple GPUs and you want
AgentKernelArena to keep all of them busy. The parallel runner starts one
long-lived Docker worker container per GPU. Each worker sees exactly one GPU,
claims one task at a time from a shared queue, runs it to completion, and then
claims the next available task.

This keeps the serial `make docker-run` workflow unchanged while adding a
host-side scheduler for multi-GPU servers.

## Start a parallel run

Select a run configuration, then list the GPUs to use with `GPU_IDS`:

```bash
CONFIG_PATH=example_configs/benchmark_cursor_mi355x.yaml
make docker-parallel-run CONFIG="$CONFIG_PATH" GPU_IDS=0,1,2,3
```

On an 8-GPU server:

```bash
make docker-parallel-run CONFIG="$CONFIG_PATH" GPU_IDS=0,1,2,3,4,5,6,7
```

If `GPU_IDS` is omitted, the runner discovers GPUs with `rocm-smi --showid` and
starts one worker for each discovered GPU:

```bash
make docker-parallel-run CONFIG="$CONFIG_PATH"
```

Use `RUN_ARGS` the same way as `docker-run`:

```bash
make docker-parallel-run \
  CONFIG="$CONFIG_PATH" \
  GPU_IDS=0,1,2,3,4,5,6,7 \
  RUN_ARGS="--run-suffix cursor_parallel8"
```

## How scheduling works

The runner creates a shared queue in the run directory:

```text
workspace_<gpu>_<agent>/
└── run_<timestamp>[_suffix]/
    └── .parallel/
        ├── pending/
        ├── running/
        ├── done/
        └── failed/
```

Each task has one descriptor file. Workers claim tasks by atomically renaming a
descriptor from `pending/` to `running/`. After the task finishes, the descriptor
moves to `done/` or `failed/`. This prevents duplicate task claims while allowing
faster tasks to free a GPU and immediately claim more work.

After all workers exit, the runner starts a single post-processing container to
aggregate results for the whole run.

## GPU isolation

Each worker container is assigned one host GPU. The Docker runner configures the
container so that ROCm masks to the host GPU, while framework code inside the
masked container sees that GPU as logical device `0`:

```text
AGENT_KERNEL_ARENA_HOST_GPU_ID=<host_gpu_id>
ROCR_VISIBLE_DEVICES=<host_gpu_id>
HIP_VISIBLE_DEVICES=0
CUDA_VISIBLE_DEVICES=0
GPU_DEVICE_ORDINAL=0
```

The runner also gives each worker its own temporary `HOME`, `CODEX_HOME`, and
cache directories for Torch extensions, Triton, MIOpen, Matplotlib, and agent
state. Only the agent selected by the run configuration is provisioned: its host
CLI and authentication/configuration state are mounted read-only and copied into
the worker-local home before the agent starts.

## Resume a parallel run

Resume works the same way as serial runs. Completed tasks are skipped, and
unfinished tasks are returned to the pending queue.

```bash
make docker-parallel-run \
  CONFIG="$CONFIG_PATH" \
  GPU_IDS=0,1,2,3,4,5,6,7 \
  RUN_ARGS="--resume-run run_20260702_041903_parallel8"
```

You can also resume the most recent run:

```bash
make docker-parallel-run CONFIG="$CONFIG_PATH" GPU_IDS=0,1 RUN_ARGS="--resume-latest"
```

Completion is detected from the per-task workspace:

- Normal optimization agents: `task_result.yaml`
- `task_validator`: `validation_report.yaml`

## Run the task validator in parallel

The `task_validator` agent uses the same worker queue. This is useful when
validating many tasks before a release or shared experiment. Save this example
as `config_parallel_validator_mi355x.yaml`:

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

The final `validation_summary.yaml` is written once after all workers complete.

## Failure behavior

If one task fails, its descriptor moves to `.parallel/failed/` and that worker
continues with the next task. Other GPU workers keep running. At the end of the
run, the final command returns nonzero if any descriptor is left in `failed/`,
`pending/`, or `running/`.

Task-level failures are different from runner failures. For example, an agent can
successfully produce `task_result.yaml` with `pass_correctness: false`; that task
is still considered completed by the scheduler and will appear in the aggregate
report.
