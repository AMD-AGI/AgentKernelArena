---
myst:
    html_meta:
        "description": "Configure AgentKernelArena agents for controlled A/B experiments and RL-ready GPU kernel optimization runs."
        "keywords": "AgentKernelArena, agents, models, Claude Code, Cursor, Codex, GEAK, A/B testing, agent RL, ROCm, GPU"
---

# Configure agents and models in AgentKernelArena

AgentKernelArena evaluates one agent per run. The agent is selected by the
`agent.template` field in the chosen run configuration. This topic lists the
supported agents, explains how models and providers are configured, and
describes how to use the arena for A/B testing.

## Supported agents

The following agents are available.

| `agent.template` | Description |
| --- | --- |
| `cursor` | Cursor Agent CLI |
| `claude_code` | Anthropic Claude Code CLI |
| `codex` | OpenAI Codex CLI |
| `geak_v3` | Specialized GEAK integration for HIP optimization |
| `geak_v3_triton` | Specialized GEAK integration for Triton optimization |
| `mini_swe_triton` | mini-swe-agent-based Triton optimization |
| `task_validator` | Task quality validator; does not optimize kernels (see [Validate tasks](task-validator.md)) |

Select one in a run configuration:

```yaml
agent:
  template: claude_code
```

Each agent lives under `agents/<agent_name>/` and is registered into a shared
registry, so the framework loads only the agent you select.

The Cursor, Claude Code, and Codex integrations reuse their host CLI login
state. Specialized integrations have additional setup and configuration under
their respective `agents/<agent_name>/` directories.

## Models, providers, and agent settings

AgentKernelArena has no shared model/provider field in the run configuration.
The selected integration controls its own model, provider, authentication,
effort, timeout, and iteration settings through its CLI and
`agents/<agent_name>/agent_config.yaml`.

For Cursor, Claude Code, and Codex, authenticate with the host CLI. A normal run
preflights only the selected CLI. When the config selects one of these
first-class integrations (or `task_validator`), select the run configuration
first and check its CLI/backend:

```bash
CONFIG_PATH=example_configs/quickstart_claude_mi300.yaml
make docker-check-agents CONFIG="$CONFIG_PATH"
```

Use `AGENTS=<comma-separated names>` for an explicit subset or `AGENTS=all` for
all three first-class CLIs and login states. Specialized integrations are not
handled by this command; their README files document their own dependencies,
API keys, and endpoint configuration.

`make vllm` starts an OpenAI-compatible local endpoint on port `30001`, but it
does not automatically reconfigure an agent. Point the selected integration at
that endpoint using the integration's own provider/base-URL mechanism.

## A/B testing and ablation studies

AgentKernelArena is designed to test changes to agent behavior: a model, Model
Context Protocol (MCP) server, skill, prompt strategy, tool integration, memory
strategy, or policy.

Run the same task set twice, once with the capability enabled and once
without, then compare the standardized scores:

```bash
CONFIG_PATH=example_configs/quickstart_claude_mi300.yaml

# Baseline
make docker-run CONFIG="$CONFIG_PATH" RUN_ARGS="--run-suffix baseline"

# With the new capability enabled in the agent configuration
make docker-run CONFIG="$CONFIG_PATH" RUN_ARGS="--run-suffix with_capability"
```

Both runs land in the same workspace directory with distinct run names, so the
[visualization dashboard](visualization.md) can show them side-by-side. Hold
every non-treatment factor constant—including tasks, hardware, environment, and
scoring—and repeat matched trials when agent behavior is stochastic. The
observed deltas estimate the effect of the capability under test.

You can also generate a text comparison directly:

```bash
python3 compare_runs.py \
  workspace_MI300_claude_code/run_<timestamp>_baseline \
  workspace_MI300_claude_code/run_<timestamp>_with_capability
```

The resulting `task_result.yaml` files expose compilation, correctness, timing,
speedup, and score fields that an external RL system can consume as reward
signals. AgentKernelArena does not itself update a policy.

## Add a new agent

To integrate a custom agent:

1. Create `agents/<your_agent>/` with a
   `launch_agent(eval_config, task_config_dir, workspace)` function decorated
   with `@register_agent("<your_agent>")`.
2. Add an entry to the `AgentType` enum and an import branch in
   `src/module_registration.py`.
3. Wire the agent into the prompt builder and post-processing handler in
   `src/module_registration.py` if it needs the standard behavior.

See the development section of the repository [`README.md`](https://github.com/AMD-AGI/AgentKernelArena/blob/main/README.md#development) for the full template.
