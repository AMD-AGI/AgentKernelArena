---
myst:
    html_meta:
        "description": "Install the Docker-first AgentKernelArena experimentation environment, pinned ROCm/SGLang images, and supported agent integrations."
        "keywords": "AgentKernelArena, install, Docker, ROCm, SGLang, AMD GPU, HIP, PyTorch, agent CLI"
---

# Install AgentKernelArena

AgentKernelArena runs controlled agent experiments against GPU kernel tasks on
an AMD GPU. Docker is the supported workflow: each experiment runs inside the
GPU-architecture-specific runtime image and bind-mounts the required local agent
CLI. Authentication uses the integration's host login state or provider environment
variables.

## Prerequisites

Verify that you have the following before continuing.

- **A Linux host with an AMD GPU and the ROCm kernel driver installed:**
  The runner passes the GPU into the container through `/dev/kfd` and
  `/dev/dri`; both device files must exist on your host. If they are missing,
  install the driver from the
  [ROCm installation guide](https://rocm.docs.amd.com/en/latest/deploy/linux/index.html).

- **Docker Engine, accessible without `sudo`:**
  Every experiment runs inside a container. The Makefile targets call Docker
  directly, so your user must be able to reach the Docker daemon without
  elevated privileges. Follow the
  [Docker post-installation steps](https://docs.docker.com/engine/install/linux-postinstall/)
  to add your user to the `docker` group if needed.

- You need **Git** to clone the repository. Install it from
  [git-scm.com](https://git-scm.com/downloads) or your distribution's package
  manager.

- **An installed and authenticated agent CLI:**
  The runner provisions only the agent selected by your run configuration; it
  cannot install or authenticate the CLI for you. Install and authenticate
  before running an agent check or experiment. See
  [Install agent CLIs](#install-agent-clis) below, then
  [Configure agents and models](../how-to/agents.md) for model and
  provider setup.

- **Node.js 22+ and npm** *(required only for npm-installed agent CLIs):*
  Claude Code supports npm as an alternative to its native installer. DeepSeek
  Harness requires a dedicated Node.js 24 prefix with both `bin/node` and
  `bin/dsh`. If you plan to use either, install Node.js from
  [nodejs.org](https://nodejs.org/). See the
  [DeepSeek Harness README](https://github.com/AMD-AGI/AgentKernelArena/blob/main/agents/deepseek_harness/README.md)
  for its specific Node.js prefix setup.

## Set up the Docker runner

Clone the repository and verify the container runtime and GPU:

```bash
git clone https://github.com/AMD-AGI/AgentKernelArena.git
cd AgentKernelArena

make docker-smoke
```

The runner mounts the selected integration's host login state or forwards its
provider environment variables — it does not copy credentials into an image.

Before starting an experiment, install and authenticate the agent selected by
your configuration. The next section covers the supported host CLIs.

**Optional:** Open an interactive shell in the same environment:

```bash
make docker-shell
```

On a Slurm or Spur cluster where the login node has no GPU or Docker daemon, use
the scheduler wrapper instead. See [Run on Slurm or Spur GPU nodes](../how-to/slurm-run.md).

### Select an image

The runner selects a default image based on the GPU architecture. To use a
different image, set `docker_image` in your run YAML, or set `AKA_DOCKER_IMAGE`
or `AKA_DOCKER_IMAGE_<ARCH>` in the environment. Use an immutable
`repository@sha256:...` reference when pinning an image for reproducibility.

Image selection priority: `AKA_DOCKER_IMAGE`, `AKA_DOCKER_IMAGE_<ARCH>`, the
config's `docker_image`, then the architecture default. Selecting an image does
not change the GPU architecture; that still comes from `target_gpu_model` or
`AKA_GPU_ARCH`. Config image overrides also disable automatic RDNA4 builds.
Invalid values fail before a container is launched.

Always start a fresh run after changing images. Runs record the selected image
in materialization and baseline state, and resume rejects a changed runtime.
Measure baseline and candidate in the same runtime.

Some tasks require a dedicated runtime — for example, tasks that materialize vLLM
source directories. Set their documented image explicitly through `docker_image`;
the MI355X default does not supply every task's external dependencies. Optional
evaluation tools have their own [scoring-image compatibility requirements](../how-to/use-evaluation-tools.md).

:::{note}
The MI355X runner routes `rocprofv3` to the same SDK installation used by
PyTorch workloads. `rocprof-compute` is not bundled in this image. See the
[runtime guide](../reference/mi355x-runtime.md#sdk-and-profiling) for SDK
library selection and profiler availability.
:::

**Optional — roll back to the previous MI355X runtime:** Set
`AKA_DOCKER_IMAGE_GFX950` to the old digest and start a fresh run:

```bash
AKA_DOCKER_IMAGE_GFX950=lmsysorg/sglang-rocm@sha256:b435b508b5aa696abb25c909341ce73e41574c4271cf716bed72418dcea86b78 \
  make docker-run CONFIG=example_configs/quickstart_claude_mi355x.yaml
```

Alternatively, select `GFX950_V0514_IMMUTABLE_IMAGE` from the
[runner image definitions](https://github.com/AMD-AGI/AgentKernelArena/blob/main/src/scripts/docker_benchmark.sh)
via `docker_image` or `AKA_DOCKER_IMAGE`. Evaluation tools also need the
matching repository revision and sidecar set; the ROCm 10 tool profile
deliberately rejects the old scoring image.

### Set up RDNA4 (`gfx1201`)

The runner automatically builds the default RDNA4 image on first use if it is
missing. Run the smoke check on the GPU host:

```bash
make docker-smoke
```

After installing and authenticating Claude Code, verify the agent and run a task:

```bash
CONFIG_PATH=example_configs/quickstart_claude_rdna4.yaml
make docker-check-agents CONFIG="$CONFIG_PATH"
make docker-run CONFIG="$CONFIG_PATH"
```

**Optional:** To prebuild or rebuild the image before running:

```bash
make docker-build-rdna4
```

Explicit image overrides disable automatic builds. The smoke check requires
`rocprofv3` on `gfx1201`. A successful smoke establishes runtime availability,
not support for every task or profiler analysis of an optimized candidate. See
the [image recipe and limitations](https://github.com/AMD-AGI/AgentKernelArena/blob/main/docker/rdna4/README.md)
for the pinned base, non-root runtime layout, and task coverage.

## Install agent CLIs

Install whichever agent you plan to run. Claude Code recommends its native
installer; npm is also supported by the Docker runner:

```bash
# Recommended native installation on Linux/macOS/WSL:
curl -fsSL https://claude.ai/install.sh | bash
claude --version

# Alternative npm installation (requires Node.js 22+):
# node --version
# npm --version
# npm install -g @anthropic-ai/claude-code

# Authenticate on the host after either installation, then exit Claude Code.
claude
claude auth status

# Alternatively, install Cursor Agent:
# make install-cursor-agent

# For Codex, follow its official CLI instructions and ensure `codex` is on PATH.
```

Installing an agent CLI is not enough: authenticate it on the host before an
agent check or experiment. The Docker runner supports both Claude Code's native
installation and its alternative npm installation. See the
[official Claude Code setup guide](https://code.claude.com/docs/en/installation)
for current installation alternatives.

For DeepSeek Harness, follow the [pinned CLI installation and API-key
setup](https://github.com/AMD-AGI/AgentKernelArena/blob/main/agents/deepseek_harness/README.md#install-and-configure). Its Node
prefix must contain both `bin/node` and `bin/dsh`; select that prefix with
`AKA_NODE_PREFIX` when necessary. Host `~/.dsh` state is not mounted.

Specialized integrations require additional runtime setup. Review
[GEAK](https://github.com/AMD-AGI/AgentKernelArena/blob/main/agents/geak/README.md) or
[Forge](https://github.com/AMD-AGI/AgentKernelArena/blob/main/agents/forge/README.md) before selecting one.

## Choose an example configuration

Choose the configuration that matches the physical GPU and installed agent.
The quickstart configurations each run one GELU task. For a longer Cursor Agent
run, use the curated task selection in
[`benchmark_cursor_mi355x.yaml`](https://github.com/AMD-AGI/AgentKernelArena/blob/main/example_configs/benchmark_cursor_mi355x.yaml).

| Configuration | Purpose |
| --- | --- |
| `example_configs/quickstart_claude_mi300.yaml` | Generic MI300-series (`gfx942`) Claude Code GELU task; inspect the physical model before tuning. |
| `example_configs/quickstart_claude_mi300x.yaml` | Claude Code GELU task with the MI300X hardware profile (`gfx942`). |
| `example_configs/quickstart_claude_mi325x.yaml` | Claude Code GELU task with the MI325X hardware profile (`gfx942`). |
| `example_configs/quickstart_claude_mi300a.yaml` | Claude Code GELU task with the MI300A hardware profile (`gfx942`). |
| `example_configs/quickstart_claude_mi355x.yaml` | First Claude Code run on MI355X (`gfx950`). |
| `example_configs/quickstart_claude_rdna4.yaml` | First Claude Code run on RDNA4 (`gfx1201`); builds the default runtime on first use if missing. |
| `example_configs/quickstart_deepseek_harness_mi300.yaml` | First DeepSeek Harness run on MI300/MI300X (`gfx942`). |
| `example_configs/quickstart_deepseek_harness_mi355x.yaml` | First DeepSeek Harness run on MI355X (`gfx950`). |
| `example_configs/benchmark_cursor_mi355x.yaml` | Curated Cursor Agent benchmark on MI355X; requires an installed and authenticated Cursor Agent CLI. |

The default `make docker-run` configuration is the generic MI300-series quickstart.
On another GPU, always pass the matching `CONFIG`; for MI355X, use
`example_configs/quickstart_claude_mi355x.yaml`.

Select one quickstart configuration in the current shell:

```bash
# MI300/MI300X:
CONFIG_PATH=example_configs/quickstart_claude_mi300.yaml

# For MI355X instead:
# CONFIG_PATH=example_configs/quickstart_claude_mi355x.yaml
```

Verify the agent selected by that configuration, then run the single task:

```bash
make docker-check-agents CONFIG="$CONFIG_PATH"
make docker-run CONFIG="$CONFIG_PATH"
```

To create another experiment, copy the nearest example and edit the copy rather
than changing a shared example in place:

```bash
cp "$CONFIG_PATH" my_experiment.yaml
```

## Install FlyDSL (optional)

`flydsl2flydsl`, `torch2flydsl`, `triton2flydsl`, and `operator2flydsl` tasks
need the `flydsl` package inside the container. The selected image may already
ship it (`make docker-smoke` prints `flydsl=ok <version>` when present). If
yours does not, install it once into the container's persistent pip user-base:

```bash
make docker-setup-flydsl
```

This is a no-op when the image already provides FlyDSL.

## Configure authentication and providers

Cursor, Claude Code, and Codex reuse their host CLI authentication. A normal run
preflights only its selected CLI. For another config that selects one of these
CLIs—or `task_validator`, which resolves to its configured backend—check the
same CLI without starting a task by passing that run config:

```bash
make docker-check-agents CONFIG=my_experiment.yaml

# Optional overrides:
make docker-check-agents AGENTS=claude_code,codex
make docker-check-agents AGENTS=all
```

`AGENTS=all` is the explicit strict check for Cursor, Claude Code, and Codex.

Codex uses the host's `~/.codex` directory by default. To use a separate, small
authentication/configuration directory, set `AKA_CODEX_AUTH_DIR` before an agent
check or run:

```bash
AKA_CODEX_AUTH_DIR="$HOME/.config/aka-codex-auth" \
  make docker-check-agents CONFIG=my_experiment.yaml
```

The directory must already exist, be readable, and contain the authentication
and configuration needed by the selected Codex CLI. It can omit sessions,
worktrees, and package history. With worker-home isolation enabled, the runner
mounts this source read-only and copies its contents into the worker's writable
temporary home; worker updates do not write back to the source. Without
isolation, it replaces the usual writable `~/.codex` mount. Native Codex packages
remain mounted from the host installation, independently of this override.
The normal CLI authentication check still runs. Leaving `AKA_CODEX_AUTH_DIR`
unset preserves the existing mounts and copying behavior.

DeepSeek Harness is opt-in: select its run config or pass
`AGENTS=deepseek_harness`. Export `DEEPSEEK_API_KEY` before the check/run; the
runner forwards it by environment-variable name only for DeepSeek runs.
Its check verifies CLI version and credential presence without contacting the
provider. Model and endpoint settings live in
[`agents/deepseek_harness/agent_config.yaml`](https://github.com/AMD-AGI/AgentKernelArena/blob/main/agents/deepseek_harness/agent_config.yaml).

Specialized integrations read credentials and provider endpoints from their
own environment/configuration; their documentation describes the additional
setup and checks. There is no shared provider field in the root run
configuration.

## Verify the installation

The quickstart run above confirms that the framework, GPU, and agent CLI work
together. It creates a timestamped workspace directory
(`workspace_<gpu>_<agent>/run_<timestamp>/`), logs to `logs/`, and writes a
`task_result.yaml` for the task.

To run across multiple GPUs, list host GPU IDs or omit `GPU_IDS` to discover
them with `rocm-smi --showid`:

```bash
make docker-parallel-run CONFIG="$CONFIG_PATH" GPU_IDS=0,1,2,3
```

See [Run an experiment](../how-to/run-evaluation.md) for the full workflow.
