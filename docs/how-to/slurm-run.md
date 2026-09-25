---
myst:
    html_meta:
        "description": "Run AgentKernelArena from a GPU-less login node by allocating MI355X GPUs with Slurm/Spur and launching Docker on the compute node."
        "keywords": "AgentKernelArena, Slurm, Spur, MI355X, login node, Docker, multi-GPU"
---

# Run on Slurm/Spur GPU nodes

Use the Slurm/Spur runner when the development host is a GPU-less login node
and Docker is available only on allocated compute nodes. The wrapper requests
the GPU resources first, then runs the existing Docker workflow on that node:

```text
login node -> srun/sbatch allocation -> compute-node Docker -> main.py
```

The runner does not use Slurm's `--container-image` option. Docker remains the
runtime backend, preserving the pinned images, agent mounts, preflight checks,
and task behavior used by the directly attached GPU-host workflow.

## Prerequisites

- `srun` and `sbatch` are available on the login node.
- The repository and agent login state are on a filesystem visible from the
  compute nodes at the same absolute paths.
- For DeepSeek Harness, its dedicated Node prefix must also be visible there,
  and `DEEPSEEK_API_KEY` must be exported in the submitting shell.
- Docker and the ROCm devices are available after allocation.
- The current user can use the compute-node Docker daemon.
- The configured GPU type and runtime image match the allocated hardware.

The defaults target the `amd-spur` partition and MI355X (`gfx950`) nodes.

## Develop with one GPU

Run a short runtime check before starting an experiment:

```bash
make slurm-smoke
```

Check the CLI and authentication selected by a run config:

```bash
make slurm-check-agents CONFIG=config_codex_mi355x_spur.yaml
```

Run synchronously and keep output attached to the terminal:

```bash
make slurm-run \
  CONFIG=config_codex_mi355x_spur.yaml \
  RUN_ARGS="--run-suffix dev"
```

For an interactive shell in the same GPU runtime:

```bash
make slurm-shell
```

These commands request one GPU by default. Spur reports the allocated physical
device IDs in `SPUR_JOB_GPUS`; the wrapper forwards the selected ID as
`ROCR_VISIBLE_DEVICES`, while the application sees logical GPU `0`.

## Run eight workers on one node

Validate all eight GPU workers without starting an experiment:

```bash
make slurm-parallel-smoke
```

Run a multi-task configuration synchronously:

```bash
make slurm-parallel-run \
  CONFIG=example_configs/benchmark_cursor_mi355x.yaml \
  RUN_ARGS="--run-suffix parallel8"
```

The wrapper requests one node with eight MI355X GPUs. The existing parallel
queue starts one Docker worker per allocated physical GPU; every worker sees
its assigned device as logical GPU `0`.

The Slurm wrapper currently supports one node. It does not distribute the
shared queue across multiple nodes.

## Submit batch jobs

Submit a one-GPU run:

```bash
make slurm-submit \
  CONFIG=config_codex_mi355x_spur.yaml \
  RUN_ARGS="--run-suffix batch"
```

Submit an eight-GPU run:

```bash
make slurm-parallel-submit \
  CONFIG=example_configs/benchmark_cursor_mi355x.yaml \
  RUN_ARGS="--run-suffix parallel8"
```

`sbatch --parsable` prints the job ID. Scheduler stdout and stderr are written
under `logs/slurm/` by default. Use `squeue` and the cluster's normal
cancellation command to manage the submitted job.

## Resource overrides

Override Make variables on the command line:

```bash
make slurm-run \
  CONFIG=config_codex_mi355x_spur.yaml \
  SLURM_TIME=02:00:00 \
  SLURM_CPUS=24 \
  SLURM_MEM=128G
```

The available settings are:

| Make variable | Default | Purpose |
| --- | --- | --- |
| `SLURM_PARTITION` | `amd-spur` | Target partition |
| `SLURM_GPU_TYPE` | `mi355x` | Scheduler GPU type |
| `SLURM_GPU_ARCH` | `gfx950` | Container runtime architecture |
| `SLURM_GPU_COUNT` | `1` | GPU count for development commands |
| `SLURM_PARALLEL_GPU_COUNT` | `8` | GPU count / worker count for parallel commands |
| `SLURM_CPUS`, `SLURM_MEM`, `SLURM_TIME` | `16`, `64G`, `04:00:00` | One-GPU resources |
| `SLURM_PARALLEL_CPUS`, `SLURM_PARALLEL_MEM`, `SLURM_PARALLEL_TIME` | `64`, `512G`, `1-00:00:00` | Eight-GPU resources |
| `SLURM_ACCOUNT`, `SLURM_QOS` | empty | Optional accounting settings |
| `SLURM_NODELIST` | empty | Optional node restriction |
| `SLURM_EXCLUSIVE` | `0` | Set to `1` for exclusive-node benchmarking |
| `SLURM_LOG_DIR` | `logs/slurm` | Batch stdout/stderr directory |

## Runtime and authentication behavior

Spur exposes the allocation through `SPUR_JOB_GPUS` and
`ROCR_VISIBLE_DEVICES`, but Docker does not inherit those variables
automatically. The wrapper passes an explicit physical device mask to every
container. It also omits `/dev/mem`, which Spur's Docker authorization plugin
does not permit; AgentKernelArena tasks use `/dev/kfd` and `/dev/dri`.

Agent installation and authentication state are mounted read-only as the source
for each run. The Docker runner copies mutable state into an isolated temporary
`HOME` for each worker. Native standalone and npm-based Codex installations,
native and npm-based Claude Code installations, and native Cursor Agent
installations are supported.

Claude subscription logins contain rotating OAuth credentials. Independent
copies can become stale after another worker refreshes the same login. A
successful `claude auth status` only confirms local login state; it does not
prove that the next inference or token refresh will succeed. An agent startup
failure is not a completed optimization, even when the unchanged starting
implementation passes the final numerical checks.

For unattended or parallel Claude/GEAK runs, Claude's documented
[`claude setup-token` authentication](https://code.claude.com/docs/en/authentication#generate-a-long-lived-token)
provides a long-lived inference token through `CLAUDE_CODE_OAUTH_TOKEN`. Set it
through your credential manager before submission; never put its value in a
run YAML, shell command argument, tracked file, or report. The Docker runner
passes this environment variable by name only to containers that run Claude
Code, including GEAK, and does not require browser-login files in this mode.

For an existing browser login, `AKA_CLAUDE_AUTH_DIR` can select a separately
managed directory containing `.credentials.json`; the default remains
`~/.claude`. Under worker isolation this directory is mounted read-only, then
copied into the private worker home. Keep it outside task/source/report trees,
owned by the submitting user, with directory mode `0700` and credential mode
`0600`. This is an explicit credential source, not a refresh coordinator:
updating a seed does not update already-running copies, and Arena does not
write refreshed credentials back to the user's login. Coordinate renewal or
use setup-token for runs that outlast the current login lease. Preserve failed
run evidence and launch a new attempt after authentication is restored.

DeepSeek Harness uses a read-only Node/CLI prefix and an exported API key.
Follow [its setup guide](../../agents/deepseek_harness/README.md), including
`AKA_NODE_PREFIX` when the installation is outside the detected path. The
scheduler inherits the exported variables, and Docker forwards the key by name
only when DeepSeek is selected. Each task creates its own `DSH_HOME`; the host's
`~/.dsh` profiles and sessions are not mounted. After setup:

```bash
CONFIG_PATH=example_configs/quickstart_deepseek_harness_mi355x.yaml
make slurm-check-agents CONFIG="$CONFIG_PATH"
make slurm-run CONFIG="$CONFIG_PATH" RUN_ARGS="--run-suffix deepseek_smoke"
```

Keep the literal API key out of Make arguments, batch scripts, and job logs.
The check verifies CLI version and key presence; the run exercises API access.

Docker images are cached per compute node. The first job scheduled onto a node
may spend several minutes pulling the selected SGLang image.
