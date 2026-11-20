---
myst:
    html_meta:
        "description": "Run AgentKernelArena from a GPU-less login node by allocating MI355X GPUs with Slurm or Spur and launching Docker on the compute node."
        "keywords": "AgentKernelArena, Slurm, Spur, MI355X, login node, Docker, multi-GPU"
---

# Run AgentKernelArena on Slurm or Spur GPU nodes

Use this guide if you are working from a login node that has no GPU and no
Docker daemon — a common setup on managed HPC clusters such as AMD Spur. The
Slurm or Spur wrapper allocates a GPU compute node on your behalf, then runs the
same Docker workflow you would use on a directly attached GPU host:

```text
login node → srun/sbatch allocation → compute-node Docker → main.py
```

The wrapper does not use Slurm's `--container-image` option. Docker remains the
runtime backend, so pinned images, agent mounts, preflight checks, and task
behavior are identical to a direct run. The workflow your agent sees is unchanged.

This guide covers:

1. Verifying the runtime with a smoke check before committing to a full run
2. Running and debugging a single-GPU experiment interactively
3. Scaling to eight parallel GPU workers on one node
4. Submitting unattended batch jobs with `sbatch`
5. Overriding partition, GPU type, and resource defaults to match your cluster

Authentication is different in this workflow. Claude's OAuth credentials can
go stale when multiple parallel workers share the same login state. For
unattended or parallel Claude runs, set up a long-lived token before submitting —
see [Set up authentication](#set-up-authentication) below.
The first job scheduled onto a fresh compute node may also take several minutes
to pull the runtime image.

## Prerequisites

- **`srun` and `sbatch` available on the login node.**
  These are the Slurm commands that allocate GPU resources and launch jobs on
  compute nodes. The `make slurm-*` targets call them directly. If your cluster
  uses Spur, they are provided through the Spur interface. Verify they are on
  your `PATH` with `which srun sbatch` before continuing.

- **A shared filesystem visible from compute nodes at the same absolute paths.**
  After allocation, the wrapper runs on a compute node — not the login node. It
  must be able to reach the AgentKernelArena repository and your agent's login
  state at the same absolute paths. Most HPC clusters mount shared storage (NFS,
  Lustre, or similar) under `/home` and project directories automatically. If
  your repository is in a login-node-local path, move it to a shared mount first.

- **Docker Engine on the compute node, accessible without `sudo`.**
  The Slurm wrapper allocates the GPU node and runs the same Docker workflow used
  by the direct GPU-host setup. Docker must be installed on compute nodes and
  your user must be able to reach the daemon without elevated privileges — the
  same requirement as [the direct runner](../install/install.md#set-up-the-docker-runner).
  This is typically pre-configured on managed HPC clusters; ask your cluster
  administrator if you are unsure.

- **ROCm GPU devices exposed after allocation.**
  Once a compute node is allocated, `/dev/kfd` and `/dev/dri` must be accessible.
  The wrapper passes these devices explicitly into every container. Run
  `make slurm-smoke` to confirm the runtime and GPU are both working; device
  discovery alone does not establish a working runtime.

- **A run configuration whose GPU type and image match the allocated hardware.**
  A mismatch between the requested hardware and the runtime image causes failures
  that are hard to diagnose after the job starts. The defaults target the
  `amd-spur` partition and MI355X (`gfx950`) nodes. If you are running on
  different hardware, override `SLURM_GPU_TYPE`, `SLURM_GPU_ARCH`, and
  `SLURM_PARTITION` before running — see [Resource overrides](#resource-overrides).

- **An installed and authenticated agent CLI on the shared filesystem.**
  The Docker runner mounts your agent's login state read-only and copies it into
  each worker's isolated home directory. The CLI and its credentials must be on
  the shared filesystem so the compute node can reach them. See
  [Install agent CLIs](../install/install.md#install-agent-clis) and
  [Set up authentication](#set-up-authentication)
  for how credentials are handled during parallel runs.

- **DeepSeek Harness: additional shared-filesystem requirements** *(if applicable).*
  The dedicated Node.js prefix must be on the shared filesystem and reachable
  from the compute node. Export `DEEPSEEK_API_KEY` in the submitting shell;
  the scheduler inherits it and Docker forwards it into the container by name.
  See the [DeepSeek Harness setup guide](https://github.com/AMD-AGI/AgentKernelArena/blob/main/agents/deepseek_harness/README.md)
  for the required Node prefix layout.

## Set up authentication

Authentication requires extra care on cluster runs. Each Docker worker gets an
isolated copy of your agent's credential state; copies do not share token
refreshes with each other. For a single interactive run, an existing browser
login typically works. For parallel or unattended runs, follow the steps below
before submitting.

### Claude and GEAK

Claude subscription logins use rotating OAuth credentials. When multiple
workers start from the same credential file, one worker's token refresh can
invalidate the copies the others are holding. A successful `claude auth status`
only confirms local login state — it does not prove that the next inference or
token refresh will succeed. An agent startup failure is not a completed
optimization, even when the unchanged starting implementation passes the final
numerical checks.

For unattended or parallel runs, generate a long-lived token with
[`claude setup-token`](https://code.claude.com/docs/en/authentication#generate-a-long-lived-token)
and export it as `CLAUDE_CODE_OAUTH_TOKEN` through your credential manager
before submitting. The Docker runner passes this variable by name only to
containers running Claude Code, including GEAK. This mode does not require
browser-login files. Never put the token value in a run YAML, shell command
argument, tracked file, or report.

For an existing browser login, set `AKA_CLAUDE_AUTH_DIR` to a directory
containing `.credentials.json` to use it instead of `~/.claude`. The runner
mounts it read-only and copies it into each worker's private home. The
directory must be:

- Outside task, source, and report trees
- Owned by the submitting user
- Directory mode `0700`, credential mode `0600`

This is a credential source, not a refresh coordinator — updating it does not
affect already-running workers. For runs that outlast the current login lease,
coordinate renewal or switch to `setup-token`. Preserve failed run evidence and
launch a new attempt after authentication is restored.

### DeepSeek Harness

Export `DEEPSEEK_API_KEY` in the submitting shell before running. The scheduler
inherits it and Docker forwards it into the container by name. Set
`AKA_NODE_PREFIX` if the Node.js prefix is outside the detected path. Each task
creates its own `DSH_HOME`; `~/.dsh` profiles and sessions from the host are
not mounted.

Keep the literal API key out of Make arguments, batch scripts, and job logs.

After setup, verify the CLI and run a smoke task:

```bash
CONFIG_PATH=example_configs/quickstart_deepseek_harness_mi355x.yaml
make slurm-check-agents CONFIG="$CONFIG_PATH"
make slurm-run CONFIG="$CONFIG_PATH" RUN_ARGS="--run-suffix deepseek_smoke"
```

## Develop with one GPU

Run a short runtime check before starting an experiment:

```bash
make slurm-smoke
```

The smoke check executes GPU tensor arithmetic, synchronizes, and verifies the
copied result. Device discovery alone does not establish a working runtime.

Select a maintained config matching the installed agent and physical GPU.
These examples use the MI355X Claude Code quickstart; install and authenticate
that CLI first, or choose another config using the [agent guide](agents.md).
Check the selected CLI and authentication:

```bash
CONFIG_PATH=example_configs/quickstart_claude_mi355x.yaml
make slurm-check-agents CONFIG="$CONFIG_PATH"
```

Run synchronously and keep output attached to the terminal:

```bash
make slurm-run \
  CONFIG="$CONFIG_PATH" \
  RUN_ARGS="--run-suffix dev"
```

For an interactive shell in the same GPU runtime:

```bash
make slurm-shell
```


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

Use `sbatch` when you want to submit a job and disconnect from the terminal —
for overnight runs, multi-task benchmarks, or any run you cannot supervise
interactively. The job queues in Slurm and runs when the allocation is
granted. If you are running Claude or GEAK, set up a long-lived token first
(see [Claude and GEAK](#claude-and-geak)) — browser OAuth credentials can
expire before an unattended job completes.

Submit a one-GPU run:

```bash
make slurm-submit \
  CONFIG="$CONFIG_PATH" \
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
  CONFIG="$CONFIG_PATH" \
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
| `SLURM_PARALLEL_GPU_COUNT` | `8` | GPU count and worker count for parallel commands |
| `SLURM_CPUS`, `SLURM_MEM`, `SLURM_TIME` | `16`, `64G`, `04:00:00` | One-GPU resources |
| `SLURM_PARALLEL_CPUS`, `SLURM_PARALLEL_MEM`, `SLURM_PARALLEL_TIME` | `64`, `512G`, `1-00:00:00` | Eight-GPU resources |
| `SLURM_ACCOUNT`, `SLURM_QOS` | empty | Optional accounting settings |
| `SLURM_NODELIST` | empty | Optional node restriction |
| `SLURM_EXCLUSIVE` | `0` | Set to `1` for exclusive-node benchmarking |
| `SLURM_LOG_DIR` | `logs/slurm` | Batch stdout/stderr directory |

## How the runner works

Spur exposes the GPU allocation through `SPUR_JOB_GPUS` and
`ROCR_VISIBLE_DEVICES`, but Docker does not inherit those variables
automatically. The wrapper reads them and passes an explicit physical device
mask to every container; each application sees its assigned device as logical
GPU `0`. The wrapper also omits `/dev/mem`, which Spur's Docker authorization
plugin does not permit; AgentKernelArena tasks use `/dev/kfd` and `/dev/dri`.

Agent installation and authentication state are mounted read-only as the source
for each run. The Docker runner copies mutable state into an isolated temporary
`HOME` for each worker. Native standalone and npm-based Codex installations,
native and npm-based Claude Code installations, and native Cursor Agent
installations are supported.

Docker images are cached per compute node. The first job scheduled onto a node
may spend several minutes pulling the selected SGLang image.
