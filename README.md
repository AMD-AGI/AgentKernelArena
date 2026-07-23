# AgentKernelArena: An A/B Testing and RL-Ready Environment for GPU Kernel Agents

AgentKernelArena is a controlled experimentation platform for developing AI agents on real GPU kernel optimization tasks. It enables reproducible A/B testing across models, prompts, tools, and agent policies, while providing objective compilation, correctness, and performance signals that can serve as rewards for agent reinforcement learning.

## Overview

AgentKernelArena makes changes to an agent measurable. Run the same task set with a baseline and a treatment—such as a different model, prompt, MCP server, skill, tool, memory strategy, or policy—and compare the outcomes under the same execution and scoring pipeline.

The platform provides:

- **Controlled A/B experiments**: Label and compare repeated runs while holding tasks, hardware, environment, and evaluation rules constant.
- **RL-ready feedback**: Produce per-task compilation, correctness, runtime, speedup, and score signals that can be consumed as rewards by an external reinforcement-learning system.
- **Multiple agent integrations**: Run Cursor Agent, Claude Code, Codex, GEAK-based agents, mini-swe-agent-based flows, or custom agents through a shared interface.
- **Real GPU task environments**: Work with HIP, Triton, FlyDSL, PyTorch-to-kernel conversion, instruction-to-kernel generation, and repository-level optimization tasks.
- **Isolated and reproducible execution**: Give every task its own timestamped workspace and preserve logs, modified sources, and structured results.
- **Centralized evaluation**: Measure compilation, correctness, and GPU performance independently of the optimizing agent.
- **Multi-GPU scheduling**: Start one isolated Docker worker per GPU and dynamically claim tasks from a shared queue.
- **Resumable experiments**: Resume a run without repeating tasks that already produced a completion report.
- **Held-out evaluation**: Test optimized kernels on unseen shapes and measure the generalization gap.
- **Task validation and visualization**: Validate task quality with a dedicated agent and compare local run reports in a dashboard.

AgentKernelArena supplies an environment and objective reward signals; it does not currently include an RL trainer, replay buffer, or policy-update loop. Its per-task workspaces provide reproducibility and concurrent-run separation, not a security sandbox: agent processes run permissively inside a privileged container and can access mounted repository and authentication state.

## A/B Testing Workflow

Run the same configuration twice with a distinct suffix. Change only the capability under test between the two runs.

```bash
# Choose the run configuration used for both sides of the experiment.
CONFIG_PATH=example_configs/quickstart_claude_mi300.yaml

# Baseline: capability disabled
make docker-run CONFIG="$CONFIG_PATH" RUN_ARGS="--run-suffix baseline"

# Treatment: capability enabled
make docker-run CONFIG="$CONFIG_PATH" RUN_ARGS="--run-suffix treatment"
```

Compare the generated reports directly:

```bash
python3 src/tools/compare_runs.py \
  workspace_MI300_claude_code/run_<timestamp>_baseline \
  workspace_MI300_claude_code/run_<timestamp>_treatment
```

For visual comparison, build the local dashboard as described in the
[visualization module README](src/visualization/README.md).
For stochastic agents, repeat matched baseline/treatment pairs and interpret
the observed deltas together with run-to-run variance.

## Architecture

### Core Components

```text
AgentKernelArena/
├── main.py                         # Run orchestration, resume, and parallel queue
├── example_configs/                # Quickstart and curated benchmark run configs
├── src/
│   ├── module_registration.py     # Agent registration and handler selection
│   ├── preprocessing.py            # Workspace and repository setup
│   ├── prompt_builder.py           # Task prompt construction
│   ├── evaluator.py                # Compilation and correctness evaluation
│   ├── performance.py              # Baseline and optimized timing
│   ├── score.py                    # Reward/score calculation
│   ├── postprocessing.py           # Aggregate run reports
│   ├── held_out/                   # Optional generalization evaluation module
│   ├── visualization/              # Dashboard module and static frontend
│   └── tools/
│       ├── compare_runs.py         # Compare two completed experiment runs
│       └── perf/                   # Canonical performance helpers
├── agents/
│   ├── cursor/                     # Cursor Agent CLI
│   ├── claude_code/                # Claude Code CLI
│   ├── codex/                      # Codex CLI
│   ├── geak_v3/                    # GEAK HIP optimization
│   ├── geak_v3_triton/             # GEAK Triton optimization
│   ├── mini_swe_triton/            # mini-swe-agent Triton optimization
│   └── task_validator/             # Task quality validator
├── tasks/
│   ├── hip2hip/
│   ├── triton2triton/
│   ├── instruction2triton/
│   ├── torch2hip/
│   ├── torch2flydsl/
│   ├── triton2flydsl/
│   ├── flydsl2flydsl/
│   └── repository/                 # Full-repository AITER and rocPRIM tasks
└── docs/                            # Full documentation
```

### Execution Flow

1. Load the run configuration and selected agent.
2. Discover task `config.yaml` files matching the configured selectors.
3. Create an isolated task workspace, cloning an upstream repository when required.
4. Compile and measure the original implementation to establish a baseline.
5. Build the task prompt and run the selected agent inside the workspace.
6. Independently compile, check, and time the agent's modified implementation.
7. Write a structured `task_result.yaml` containing reward signals and a score.
8. Aggregate all task results into run-level CSV, JSON, and text reports.

`task_validator` follows a validation-specific path and writes `validation_report.yaml` instead of optimizing and scoring a kernel.

For multi-GPU runs, the host-side Docker runner creates a shared `.parallel/` queue under the run directory. Each long-lived worker sees one logical GPU, atomically claims one task at a time, and returns for more work until the queue is empty. Final aggregation runs once after all workers finish.

## Supported Agents

Each run selects one `agent.template`. Repeated runs can compare different agents or different configurations of the same agent.

| Template | Purpose |
| --- | --- |
| `cursor` | Cursor Agent CLI integration |
| `claude_code` | Claude Code CLI integration |
| `codex` | Codex CLI integration |
| `geak_v3` | GEAK optimization for HIP tasks |
| `geak_v3_triton` | GEAK optimization for Triton tasks |
| `mini_swe_triton` | mini-swe-agent-based Triton optimization |
| `task_validator` | Task quality validation; does not optimize kernels |

Agent-specific models, effort settings, iteration guidance, timeouts, and provider configuration live under `agents/<agent_name>/agent_config.yaml` or in the selected agent CLI. Specialized agents may require additional setup; inspect their directories and agent-specific README files where present.

## Task Environments

| Task type | Objective |
| --- | --- |
| `hip2hip` | Optimize an existing HIP implementation |
| `triton2triton` | Optimize an existing Triton implementation |
| `instruction2triton` | Implement a Triton kernel from a specification |
| `torch2hip` | Replace a PyTorch reference with a HIP implementation |
| `torch2flydsl` | Replace a PyTorch reference with a FlyDSL implementation |
| `triton2flydsl` | Translate a Triton implementation to FlyDSL |
| `flydsl2flydsl` | Optimize an existing FlyDSL implementation |
| `repository` | Optimize a target inside a cloned upstream repository |

The prompt system also recognizes `cuda2hip`; the current bundled task tree does not include a `cuda2hip/` suite.

## Installation

### Prerequisites

- A Linux host with a supported AMDGPU kernel driver; `/dev/kfd` and `/dev/dri`
  must be present
- Docker Engine; the current user must be able to access the Docker daemon
  without `sudo`
- Git
- Node.js 22+ and npm when using the alternative npm installation of Claude Code
  (or another npm-installed agent CLI)
- The GPU-specific SGLang image: `gfx942` uses `lmsysorg/sglang:v0.5.12-rocm720-mi30x`; `gfx950` uses `lmsysorg/sglang-rocm:v0.5.14-rocm720-mi35x-20260705`
- A supported agent CLI installed and logged in on the host, or the dependencies required by a specialized agent

### Setup

```bash
git clone https://github.com/AMD-AGI/AgentKernelArena.git
cd AgentKernelArena

# Verify the container runtime, GPU, and Python environment.
make docker-smoke

# Install the primary quickstart agent with Claude Code's native installer.
curl -fsSL https://claude.ai/install.sh | bash
claude --version

# Start Claude Code and complete login on the host, then exit.
claude
claude auth status

# Select the quickstart config for the physical GPU.
CONFIG_PATH=example_configs/quickstart_claude_mi300.yaml
# For MI355X instead:
# CONFIG_PATH=example_configs/quickstart_claude_mi355x.yaml

# Verify the agent selected by the config, then run one task.
make docker-check-agents CONFIG="$CONFIG_PATH"
make docker-run CONFIG="$CONFIG_PATH"

# Other options:
# make install-cursor-agent
# Install Codex using its official CLI instructions and ensure `codex` is in PATH.
```

Installing an agent CLI is not enough: authenticate it on the host before
running `docker-check-agents` or an experiment. The Docker runner supports both
Claude Code's recommended native installation and its alternative npm
installation. The npm path requires Node.js 22+ and npm. See the
[official Claude Code setup guide](https://code.claude.com/docs/en/installation)
for the current alternatives.

The repository provides three ready-to-use run configurations:

| Configuration | Purpose |
| --- | --- |
| `example_configs/quickstart_claude_mi300.yaml` | One Claude Code GELU task on MI300/MI300X (`gfx942`); use this for a first run on MI300-series hardware. |
| `example_configs/quickstart_claude_mi355x.yaml` | One Claude Code GELU task on MI355X (`gfx950`); use this for a first run on MI355X. |
| `example_configs/benchmark_cursor_mi355x.yaml` | Curated 60-task Cursor Agent benchmark on MI355X; use this for a longer benchmark only after installing and authenticating Cursor Agent. |

Running `make docker-run` without `CONFIG` uses the MI300/MI300X Claude
quickstart. On another GPU, pass the matching configuration explicitly; for
example, use `CONFIG=example_configs/quickstart_claude_mi355x.yaml` on MI355X.

FlyDSL tasks require FlyDSL in the container. The pinned image may already provide it; otherwise run:

```bash
make docker-setup-flydsl
```

Performance timing helpers are maintained in `src/tools/perf/` and materialized into run workspaces. See [src/tools/perf/README.md](src/tools/perf/README.md) before changing task timing code.

For detailed installation and compatibility information, see [docs/install/install.md](docs/install/install.md) and [docs/reference/compatibility-matrix.md](docs/reference/compatibility-matrix.md).

## Usage

### Configure an Experiment

Start from the example that matches the physical GPU and the agent you installed.
Experiment preflight intentionally fails when the configured and detected GPU
architectures differ. To create a custom experiment, copy an example and edit
the copy:

```bash
cp example_configs/quickstart_claude_mi300.yaml my_experiment.yaml
```

Run agent-specific settings such as `model`, `effort`, `max_iterations`, and
`timeout_seconds` are configured in the selected agent's `agent_config.yaml`,
not in the run configuration.

For a Cursor, Claude Code, Codex, or task-validator config, verify only the
selected first-class host CLI (the validator resolves to its configured backend):

```bash
CONFIG_PATH=my_experiment.yaml
make docker-check-agents CONFIG="$CONFIG_PATH"
```

Use `AGENTS=claude_code,codex` to check an explicit subset or `AGENTS=all` to
check Cursor, Claude Code, and Codex together. Specialized integrations such as
GEAK and mini-swe have their own dependency checks and are not handled by this
command.

### Run Serially

```bash
make docker-run CONFIG="$CONFIG_PATH"
make docker-run CONFIG="$CONFIG_PATH" RUN_ARGS="--run-suffix experiment_name"
```

### Run Across Multiple GPUs

```bash
# Explicit host GPU IDs
make docker-parallel-run CONFIG="$CONFIG_PATH" GPU_IDS=0,1,2,3

# Discover GPUs with rocm-smi --showid
make docker-parallel-run CONFIG="$CONFIG_PATH"
```

The Docker parallel path is verified for `cursor`, `claude_code`, `codex`, and
`task_validator`. Specialized GEAK/mini-swe integrations need their own
dependencies and GPU-ID configuration before they are used with isolated
workers.

### Resume a Run

```bash
make docker-run CONFIG="$CONFIG_PATH" RUN_ARGS="--resume-latest"
make docker-run CONFIG="$CONFIG_PATH" RUN_ARGS="--resume-run run_20260702_041903_experiment"

make docker-parallel-run CONFIG="$CONFIG_PATH" GPU_IDS=0,1,2,3 RUN_ARGS="--resume-latest"
```

### Select Task Groups

Task selectors are paths relative to `tasks/`. A selector can name one task or any parent directory.

```yaml
tasks:
  - hip2hip/gpumode
  - triton2triton/vllm
  - triton2triton/rocmbench
  - instruction2triton/rocmbench
  - torch2hip
  - torch2flydsl
  - triton2flydsl
  - flydsl2flydsl
  - repository/rocprim
```

## Reward and Scoring Signals

Every normal optimization task produces `task_result.yaml` with compilation and correctness status, baseline and optimized times, speedup, timing-method metadata, and the final score.

The default score is:

| Component | Points | Condition |
| --- | ---: | --- |
| Compilation | 20 | The optimized implementation compiles |
| Correctness | 100 | Correctness passes |
| Performance | `speedup_ratio × 100` | Added only when compilation and correctness pass |

Therefore:

- Compilation fails → `0`
- Compilation passes but correctness fails → `20`
- Both pass → `120 + speedup_ratio × 100`

For multi-case tasks, the evaluator prefers the explicit per-case average `speedup_ratio` rather than deriving the score from only the two aggregate timing values. The scoring policy can be replaced in `src/score.py` when an experiment needs a different reward function.

## Task Configuration

Each isolated-kernel task has a `config.yaml`. Command and source fields are lists; `task_type` is a scalar string.

```yaml
# tasks/triton2triton/vllm/triton_rms_norm/config.yaml
source_file_path:
  - source/triton_rms_norm.py

target_kernel_functions:
  - _rms_norm_kernel

compile_command:
  - python3 scripts/task_runner.py compile

correctness_command:
  - python3 scripts/task_runner.py correctness

# Optional, but required to produce a performance reward.
performance_command:
  - python3 scripts/task_runner.py performance

task_type: triton2triton

# Optional: limit a task to a GPU architecture.
platform_support:
  required_arch: gfx942
  status: active          # active | skip
  skip_reason: null

prompt:
  source_code: null
  instructions: null
  cheatsheet: null
```

Tasks marked `platform_support.status: skip`, or requiring a different GPU
architecture, are filtered before workspace creation. Omit `platform_support`
for tasks that run on every architecture.

Repository-level tasks use `task_type: repository`, `repo_url`, and `repository_language`; their source and target hints are optional. See [docs/how-to/add-task.md](docs/how-to/add-task.md) for the complete schemas.

## Development

### Add an Agent

Create `agents/your_agent/launch_agent.py` with the interface used by `main.py`:

```python
from agents import register_agent


@register_agent("your_agent")
def launch_agent(
    eval_config: dict,
    task_config_dir: str,
    workspace: str,
) -> str:
    # Build the prompt, invoke the agent in `workspace`, and return its output.
    return result
```

Then:

1. Add the name to `AgentType` in `src/module_registration.py`.
2. Add its import branch in `load_agent_launcher`.
3. Select the standard or custom prompt-building path in the launcher.
4. Add it to `load_post_processing_handler` if it should use normal run aggregation.
5. Add an `agent_config.yaml` and focused documentation for agent-specific settings.

### Add a Task

Recommended isolated-task layout:

```text
tasks/<task_type>/[<suite>/...]/<task_name>/
├── config.yaml
├── scripts/
│   └── task_runner.py
└── source/
    └── <kernel files>
```

At minimum, isolated tasks declare list-valued `source_file_path`, `target_kernel_functions`, `compile_command`, and `correctness_command`, plus a scalar `task_type`. Add `performance_command` to measure a baseline and optimized runtime.

All new tasks must pass the task validator before merging:

Save a run configuration such as `config_task_validator.yaml`:

```yaml
agent:
  template: task_validator
tasks:
  - <full-task-path-relative-to-tasks>
target_gpu_model: MI300
log_directory: logs
workspace_directory_prefix: workspace
```

```bash
make docker-run CONFIG=config_task_validator.yaml
```

The validator runs 10 checks covering schema, source files, target symbols, compilation, correctness, performance, correctness quality, self-containedness, GPU hangs, and result compatibility. Review the generated `validation_report.yaml`; `PASS` is expected, while `WARN` requires justification. See [agents/task_validator/README.md](agents/task_validator/README.md).

## Additional Tools

- [Held-out evaluation](src/held_out/README.md): Generate unseen shapes and evaluate generalization.
- [Visualization dashboard](src/visualization/README.md): Compare local experiment reports.
- [Full documentation](docs/README.md): Installation, configuration, task authoring, parallel execution, and methodology.

## Current Directions

- Improve interactive A/B comparison and experiment tracking.
- Export richer episode traces and reward data for external agent-RL pipelines.
- Expand standardized private held-out coverage.
- Support heterogeneous agent configurations within one multi-GPU experiment.
- Continue expanding task coverage across the supported kernel environments.
