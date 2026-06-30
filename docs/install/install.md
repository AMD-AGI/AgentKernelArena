---
myst:
    html_meta:
        "description": "Learn how to install AgentKernelArena with the Docker-first benchmark runner, pinned ROCm SGLang images, host agent CLIs, and API keys."
        "keywords": "AgentKernelArena, install, Docker, ROCm, SGLang, AMD GPU, HIP, PyTorch, agent CLI"
---

# Install AgentKernelArena

AgentKernelArena runs AI coding agents against GPU kernel tasks on an AMD GPU and
evaluates the results. **Docker is the only supported workflow**: the evaluator runs
inside the GPU-arch-specific Docker image and bind-mounts the local agent CLIs
plus their login state.

## Prerequisites

- **Docker**
- **AMD GPU with ROCm-compatible Docker access** — the runner mounts `/dev/kfd`,
  `/dev/dri`, and `/dev/mem` when present.
- **Benchmark image** — `gfx942` uses
  `lmsysorg/sglang:v0.5.12-rocm720-mi30x`; `gfx950` uses
  `lmsysorg/sglang:v0.5.12-rocm720-mi35x`; Navi/RDNA/Radeon
  `gfx10*`/`gfx11*`/`gfx12*` uses
  `rocm/vllm-dev:rocm7.2.1_navi_ubuntu24.04_py3.12_pytorch_2.9_vllm_0.16.0`.
  The runner selects from `target_gpu_model` for benchmark runs and from the
  visible host GPU for shell and smoke commands.
- **Git**
- At least one supported agent CLI already installed and logged in on the host. The
  Docker runner provisions the configured agent for a run. Codex, Claude Code,
  and Cursor Agent are the first-class supported CLIs. See
  [Configure agents and models](../how-to/agents.md).

## Docker runner

From the repository root, use the Docker-first Makefile targets. The runner does
not copy credentials into an image; it bind-mounts the existing host login state.

```bash
git clone https://github.com/AMD-AGI/AgentKernelArena.git
cd AgentKernelArena

# Verify the container runtime and GPU.
make docker-smoke

# Verify Codex, Claude Code, and Cursor Agent login reuse.
make docker-check-agents
```

Start an interactive shell in the same environment:

```bash
make docker-shell
```

Run an evaluation:

```bash
make docker-run CONFIG=config.yaml
make docker-run CONFIG=config.yaml RUN_ARGS="--run-suffix cursor_docker"
```

## Install agent CLIs

Install whichever agents you plan to evaluate. For example:

```bash
# Cursor Agent CLI
make install-cursor-agent

# Claude Code
npm install -g @anthropic-ai/claude-code

# Codex CLI: follow the official Codex CLI instructions, then ensure
# `codex` is available on PATH.
```

## FlyDSL tasks (optional)

`flydsl2flydsl` tasks need the `flydsl` package inside the container. Install or
upgrade FlyDSL once into the container's persistent Python package overlay:

```bash
make docker-setup-flydsl
```

By default this installs a FlyDSL `0.2.x` wheel, even if the selected Docker image
already provides an older FlyDSL. Override the requirement with
`AKA_FLYDSL_SPEC` when a task suite needs a different version.

## Configure API keys

Export the keys for the providers you will use:

```bash
export OPENAI_API_KEY="your_openai_key"
export ANTHROPIC_API_KEY="your_anthropic_key"
export OPENROUTER_API_KEY="your_openrouter_key"
```

To run against a self-hosted model instead of a hosted provider, start a local
vLLM server:

```bash
make vllm
```

This launches a `rocm/vllm` container serving a model on port `30001`. Set
`local_llm_enabled: true` in the relevant agent configuration to use it.

## Verify the installation

Run a single quick task to confirm the framework, GPU, and agent CLI all work
together. Edit `config.yaml` to select one agent and one task, then run:

```bash
make docker-run CONFIG=config.yaml
```

A successful run creates a timestamped workspace directory
(`workspace_<gpu>_<agent>/run_<timestamp>/`), logs to `logs/`, and writes a
`task_result.yaml` per task. See [Run an evaluation](../how-to/run-evaluation.md)
for the full workflow.
