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

The following prerequisites are required before running AgentKernelArena.

- **Linux host with a supported AMDGPU kernel driver:** `/dev/kfd` and
  `/dev/dri` must be present. The runner also mounts `/dev/mem` when present.
- **Docker Engine:** the current user must be able to access the Docker daemon
  without `sudo`.
- **Runtime image:** `gfx942` uses
  `lmsysorg/sglang:v0.5.12-rocm720-mi30x`; `gfx950` uses
  the digest-pinned SGLang 0.5.20 / ROCm 10 image declared in
  [`docker_benchmark.sh`](../../src/scripts/docker_benchmark.sh). The runner selects from
  `target_gpu_model` for experiment runs and from the visible host GPU for shell
  and smoke commands.
  For `gfx1201`, the runner automatically builds the default
  [RDNA4 runtime](../../docker/rdna4/README.md) on first use if it is missing.
- **Git**
- **Node.js 22+ and npm**, when using the alternative npm installation of Claude
  Code or another npm-installed agent CLI. DeepSeek Harness uses a dedicated
  Node.js 24 prefix; see [its setup guide](../../agents/deepseek_harness/README.md).
- The selected agent CLI installed and authenticated on the host. The Docker
  runner provisions only the configured agent for a normal run. Codex, Claude
  Code, Cursor Agent, and DeepSeek Harness have host-CLI checks. See
  [Configure agents and models](../how-to/agents.md).

## Docker runner

From the repository root, use the Docker-first Makefile targets. The runner does
not copy credentials into an image; it mounts the selected integration's host
login state or forwards its provider environment variables.

An optional top-level `docker_image` string in a run YAML selects the image for
that run. Prefer an immutable `repository@sha256:...` reference. Image selection
uses this priority: `AKA_DOCKER_IMAGE`, `AKA_DOCKER_IMAGE_<ARCH>`, the config's
`docker_image`, then the architecture default. Experiment GPU architecture still
comes from `target_gpu_model` or `AKA_GPU_ARCH`; agent checks use the host GPU or
`AKA_GPU_ARCH`. Selecting an image does not change the architecture.
The setting applies to preflight, agent checks, serial runs, parallel workers,
and quality-loop runs. Shell and smoke commands use environment overrides and
architecture defaults because they do not select a run config. Config image
overrides also disable automatic RDNA4 builds. Invalid explicit values fail
before a container is launched.

The [DeepSeek draft validator config](../../example_configs/task_validator_deepseek_drafts_mi355x.yaml)
pins its MI355X run to the tested ROCm 10 image. The runner supplies writable
AITER/FlyDSL caches and AITER configuration storage for that image, which is also
the MI355X default. Other GPU architecture defaults are unchanged.

Start a separate run after changing images. Ordinary runs record the selected
image reference in materialization and baseline state, and resume rejects a
different runtime. Older workspaces without that image identity also require a
fresh run; their runtime cannot be verified retrospectively. Pin custom images
by digest when comparing experiments.
An image upgrade changes the compiler and runtime libraries as well as ROCm;
measure baseline and candidate in the same runtime and keep older results
associated with their original image.

To roll back, select `GFX950_V0514_IMMUTABLE_IMAGE` from the
[runner image definitions](../../src/scripts/docker_benchmark.sh) with
`docker_image` or `AKA_DOCKER_IMAGE`, and start a fresh run. Evaluation tools
also need the matching repository revision and sidecar set; the ROCm 10 tool
profile deliberately rejects the old scoring image.

The MI355X runner selects the SDK core libraries for workloads and routes
`rocprofv3` to that same installation used by PyTorch.
The image's original console entrypoint chooses a separate developer tree;
loading both trees in a Python workload can abort with duplicate LLVM option
registration. The wrapper preserves profiler arguments and GPU selection.
`rocprof-compute` is an optional package and is not bundled in this image; see
the [runtime guide](../reference/mi355x-runtime.md#sdk-and-profiling) for
SDK library selection and profiler availability boundaries.

Some image-backed tasks require a dedicated runtime, including tasks that
materialize vLLM source directories. Select their documented image explicitly
through the run config's `docker_image` field; the MI355X default does not supply
every task's external dependencies. Optional evaluation tools have their own
[scoring-image compatibility requirements](../how-to/use-evaluation-tools.md).

To reproduce a run on the previous MI355X runtime, explicitly select its digest:

```bash
AKA_DOCKER_IMAGE_GFX950=lmsysorg/sglang-rocm@sha256:b435b508b5aa696abb25c909341ce73e41574c4271cf716bed72418dcea86b78 \
  make docker-run CONFIG=example_configs/quickstart_claude_mi355x.yaml
```

```bash
git clone https://github.com/AMD-AGI/AgentKernelArena.git
cd AgentKernelArena

# Verify the container runtime and GPU.
make docker-smoke
```

Start an interactive shell in the same environment:

```bash
make docker-shell
```

On a Slurm/Spur cluster where the login node has no GPU or Docker daemon, use
the scheduler wrapper instead. It allocates the GPU node before invoking this
same Docker runtime. See [Run on Slurm/Spur GPU nodes](../how-to/slurm-run.md).

Install and authenticate the agent selected by your configuration before
starting an experiment; the next section covers the first-class host CLIs.

### RDNA4 (`gfx1201`)

The RDNA4 image normalizes the upstream Python and ROCm SDK paths so the same
runner can execute as the host user. On the GPU host, the first command builds
the default image if it is missing, then checks the runtime:

```bash
make docker-smoke
```

You can also start a task directly; its preflight prepares the same image.
After installing and authenticating Claude Code, use:

```bash
CONFIG_PATH=example_configs/quickstart_claude_rdna4.yaml
make docker-check-agents CONFIG="$CONFIG_PATH"
make docker-run CONFIG="$CONFIG_PATH"
```

Existing images are reused. `make docker-build-rdna4` remains available for
prebuilding or rebuilding after recipe changes. Explicit image overrides disable
automatic builds; see the runtime guide for custom image handling.

The smoke check requires `rocprofv3` on `gfx1201`; the CDNA profiler requirement
is unchanged. See the [image recipe and limitations](../../docker/rdna4/README.md)
for its pinned base, non-root runtime layout, and task coverage. A successful
smoke establishes runtime availability, not support for every task or profiler
analysis of an optimized candidate.

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
setup](../../agents/deepseek_harness/README.md#install-and-configure). Its Node
prefix must contain both `bin/node` and `bin/dsh`; select that prefix with
`AKA_NODE_PREFIX` when necessary. Host `~/.dsh` state is not mounted.

Specialized integrations require additional runtime setup. Review
[GEAK](../../agents/geak/README.md) or
[Forge](../../agents/forge/README.md) before selecting one.

## Choose an example configuration

Choose the configuration that matches the physical GPU and installed agent.
The quickstart configurations each run one GELU task; the benchmark
configuration is a longer 54-task Cursor Agent run.

| Configuration | Purpose |
| --- | --- |
| `example_configs/quickstart_claude_mi300.yaml` | First Claude Code run on MI300/MI300X (`gfx942`). |
| `example_configs/quickstart_claude_mi355x.yaml` | First Claude Code run on MI355X (`gfx950`). |
| `example_configs/quickstart_claude_rdna4.yaml` | First Claude Code run on RDNA4 (`gfx1201`); builds the default runtime on first use if missing. |
| `example_configs/quickstart_deepseek_harness_mi300.yaml` | First DeepSeek Harness run on MI300/MI300X (`gfx942`). |
| `example_configs/quickstart_deepseek_harness_mi355x.yaml` | First DeepSeek Harness run on MI355X (`gfx950`). |
| `example_configs/benchmark_cursor_mi355x.yaml` | Curated 54-task Cursor Agent benchmark on MI355X; requires an installed and authenticated Cursor Agent CLI. |

The default `make docker-run` configuration is the MI300/MI300X quickstart.
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

## FlyDSL tasks (optional)

`flydsl2flydsl`, `torch2flydsl`, `triton2flydsl`, and `operator2flydsl` tasks
need the `flydsl` package inside the container. The selected image may already
ship it (`make docker-smoke` prints `flydsl=ok <version>` when present). If
yours does not, install it once into the container's persistent pip user-base:

```bash
make docker-setup-flydsl
```

This is a no-op when the image already provides FlyDSL.

## Pinned AITER and FlyDSL on SGLang 0.5.20

For the fixed-workload AITER setup, build from the public **SGLang 0.5.20 /
ROCm 7.2** image below. It contains Torch `2.11.0+rocm7.2`, FlyDSL `0.3.2`
and an older AITER revision. The recipe installs upstream AITER
`2cb3d17589b9253ea3b3ef5222fc7f810a8f3ef0`, its pinned Composable Kernel
submodule `96afae1d54fc79c26857de7e1683b0dde1ff3c6f`, and FlyDSL `0.4.0`.
It uses public sources, so it does not require access to a locally retained
research image. Select this base explicitly; the runner's `gfx950` default is
a different SGLang 0.5.20 / ROCm 10 image.

Save the following as `Dockerfile.aiter` in a separate build directory:

```dockerfile
FROM lmsysorg/sglang@sha256:3a78acc9d6c191f1a12c7c67631657580f06af3af562c9ee6d88282a71ec5e96
USER root
ENV AITER_USE_SYSTEM_TRITON=1 PREBUILD_KERNELS=0
WORKDIR /opt
RUN git clone https://github.com/ROCm/aiter.git /opt/aka-aiter-src \
    && git -C /opt/aka-aiter-src checkout --detach 2cb3d17589b9253ea3b3ef5222fc7f810a8f3ef0 \
    && git -C /opt/aka-aiter-src submodule update --init --recursive \
    && test "$(git -C /opt/aka-aiter-src/3rdparty/composable_kernel rev-parse HEAD)" = \
       96afae1d54fc79c26857de7e1683b0dde1ff3c6f
# Explicit build/runtime prerequisites; keep the base ROCm Torch and Triton.
RUN /opt/venv/bin/python -m pip install --no-cache-dir --no-deps --no-build-isolation \
    setuptools==79.0.1 setuptools-scm==10.2.3 vcs-versioning==2.4.1 \
    wheel==0.48.0 packaging==26.2 ninja==1.13.2 pybind11==3.0.4 \
    psutil==7.2.2 pandas==3.0.3 einops==0.8.2 flydsl==0.4.0
# Replace the base image's older editable AITER registration.
RUN /opt/venv/bin/python -m pip uninstall -y amd-aiter aiter \
    && /opt/venv/bin/python -m pip install --no-cache-dir --no-deps --no-build-isolation \
       --editable /opt/aka-aiter-src
# Fresh writable caches for the host UID used by the runner.
RUN rm -rf /tmp/aiter_configs && install -d -m 1777 /tmp/aiter_configs
ENV AITER_JIT_DIR=/tmp/aka-aiter-jit \
    FLYDSL_RUNTIME_CACHE_DIR=/tmp/aka-flydsl-runtime
WORKDIR /
RUN /opt/venv/bin/python -c 'from importlib.util import find_spec; from importlib.metadata import version; from pathlib import Path; assert Path(find_spec("aiter").origin).resolve() == Path("/opt/aka-aiter-src/aiter/__init__.py"); assert version("flydsl") == "0.4.0"; assert version("torch") == "2.11.0+rocm7.2"; assert version("triton") == "3.7.0+amd.rocm7.2.0.git89002410"'
```

These prerequisites cover the pinned AITER `pyproject.toml` and `setup.py`,
including `vcs-versioning`; setuptools remains below Torch's `<82` limit.
`--no-deps`, `--no-build-isolation` and `AITER_USE_SYSTEM_TRITON=1` preserve the
base Torch/Triton installation. `PREBUILD_KERNELS=0` defers native kernel builds
to later use. The source-origin check runs outside either AITER source tree.
The cache setup is explicit because the runner's automatic cache overrides
apply only to its known image references.

Build the image, then select it through the existing runner from the
AgentKernelArena checkout:

```bash
docker build --tag aka-aiter:2cb3d175-flydsl0.4.0 - < Dockerfile.aiter
# From the AgentKernelArena repository:
AKA_DOCKER_IMAGE=aka-aiter:2cb3d175-flydsl0.4.0 make docker-shell
```

Installation belongs in the image build: `docker-shell` runs as the host UID,
while `/opt/venv` is root-owned and disables pip's `--user` mode.
`make docker-setup-flydsl` only installs a missing package; it leaves an existing
FlyDSL `0.3.2` installation unchanged. Record the built image identity and use an
immutable image reference for experiments.

**Qualification scope:** the three published `tasks/aiter_workloads/` packages
currently bind image
`sha256:1a82b6109081355b0c9b90db108349206e5fc969eed71754b96ee6919dd3fbd5`
and specific installed source, dispatch settings and native artifacts. A fresh
build above has a different runtime identity and does not satisfy those frozen
bindings. A maintainer preparing a separately versioned task/runtime must retain
fresh runtime, source, build and native-artifact bindings, then run
`task_validator` on compatible GPU hardware through
`make docker-run CONFIG=<validator-config>`. Follow
[Task definition, schema, and authoring](../how-to/add-task.md#how-to-add-or-modify-a-task)
and require a fresh framework-finalized `validation_report.yaml` with
`overall_status: PASS`. Runtime rebinding is task-authoring work; optimizing
agents continue to edit only their declared candidate paths. Package installation
alone does not establish task correctness, native dispatch or measured performance.

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
[`agents/deepseek_harness/agent_config.yaml`](../../agents/deepseek_harness/agent_config.yaml).

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
