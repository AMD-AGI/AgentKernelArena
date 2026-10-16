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

To generate tasks from external SIKL data, use the separate
[`sikl_task_builder` authoring workflow](sikl-task-builder.md). It runs before
task discovery and produces tasks for the optimization agents listed below.

The following agents are available.

| `agent.template` | Description |
| --- | --- |
| `cursor` | Cursor Agent CLI |
| `claude_code` | Anthropic Claude Code CLI |
| `codex` | OpenAI Codex CLI |
| `deepseek_harness` | DeepSeek Harness headless CLI ([setup](../../agents/deepseek_harness/README.md)) |
| `forge` | KernelForge through the shared v2 task interface |
| `geak` | GEAK Workflow engine through the shared v2 task interface |
| `task_validator` | Task quality validator; does not optimize kernels (see [Validate tasks](task-validator.md)) |

The [registry](../../src/module_registration.py) maps the legacy names
`geak_v4` to `geak` and `forge_operator2flydsl` to `forge`, with the same
configuration and post-processing. There is no separate v1 task integration. See the [GEAK](../../agents/geak/README.md) and
[Forge](../../agents/forge/README.md) guides for runtime requirements. Registry
availability alone does not establish GPU or model qualification.

Select one in a run configuration:

```yaml
agent:
  template: claude_code
```

Each agent lives under `agents/<agent_name>/` and is registered into a shared
registry, so the framework loads only the agent you select.

The supported identifiers are defined by `AgentType` in
`src/module_registration.py`. Retired templates and compatibility notes are
listed in the [release notes](../reference/release-notes.md#unreleased).

The Cursor, Claude Code, and Codex integrations reuse their host CLI login
state. Specialized integrations have additional setup and configuration under
their respective `agents/<agent_name>/` directories.

`quality_loop` is intentionally not an `agent.template`. It is a repository-level
task maintenance workflow that invokes Codex roles across the tasks selected by its
run config and publishes at most one draft PR. It never creates GitHub issues. See
[Audit and harden tasks](quality-loop.md).

## Models, providers, and agent settings

The selected integration controls model, provider, authentication, effort,
timeout, and iteration settings. Its defaults live in its CLI and
`agents/<agent_name>/agent_config.yaml`; supported fields in the run config's
`agent` mapping override those defaults. See
[CLI run-level overrides](../reference/agent-model-defaults.md#run-level-overrides)
and each specialized integration's guide for supported fields and precedence.
These settings do not belong in task configurations.

For Cursor, Claude Code, and Codex, authenticate with the host CLI. A normal run
preflights only the selected CLI. When the config selects one of these
first-class integrations (or `task_validator`), select the run configuration
first and check its CLI/backend:

```bash
CONFIG_PATH=example_configs/quickstart_claude_mi300.yaml
make docker-check-agents CONFIG="$CONFIG_PATH"
```

Use `AGENTS=<comma-separated names>` for an explicit subset or `AGENTS=all` for
the original three CLIs (Cursor, Claude Code, and Codex) and login states. GEAK and its v2 aliases resolve to
Claude Code for this check; normal runs also check the pinned GEAK engine and
SDK. Other specialized integrations have their own checks; their README files
document dependencies and provider configuration.

DeepSeek Harness supports config-selected checks and explicit
`AGENTS=deepseek_harness`. Install its pinned CLI, export `DEEPSEEK_API_KEY`,
and select one of the `quickstart_deepseek_harness_*` example configurations.
Its preflight checks version and credential presence without contacting the
provider. Each task uses a fresh Harness home and the standard Arena evaluator.
See [the integration guide](../../agents/deepseek_harness/README.md) for model,
endpoint, Docker, and session-isolation details.

DeepSeek's run-level overrides and per-invocation budget follow
[its integration guide](../../agents/deepseek_harness/README.md#install-and-configure).
It uses `reasoning_effort` rather than the other CLIs' `effort` field. Schema-v2
agent completion and independent candidate acceptance are reported separately.

## Iteration and time budgets

Cursor, Claude Code, and Codex accept `agent.budget_mode`:

| `budget_mode` | Behavior |
| --- | --- |
| `iterations` (default) | One agent session. The prompt asks for up to `max_iterations` versions, and `timeout_seconds` is the session's hard limit. Unchanged from earlier releases. |
| `time` | No iteration count. `timeout_seconds` is the agent's wall-clock budget for the task. If the agent ends its session early, Arena resumes the same session with a prompt stating the time remaining, and stops it at the original deadline. |

```yaml
agent:
  template: claude_code
  budget_mode: time
  timeout_seconds: 21600
```

The deadline is fixed when the first session starts, after the framework's
setup and baseline measurement and before candidate evaluation, so those phases
do not consume the budget. A continuation starts only while at least 10 minutes
remain, and at most 20 sessions run per task. A continuation stopped at the
deadline is not an agent failure; a first session that times out, a nonzero
exit, or a failed result still is. Arena evaluates the files delivered at the
declared paths either way.

To resume, `time` mode keeps the provider's session state: Claude Code runs
with `--session-id` instead of `--no-session-persistence`, and Codex runs
without `--ephemeral`. That state is written to the CLI's own home directory.
The shared loop is `src/tools/agent_time_budget.py`, the prompt text is
`src/prompts/agent_budget.py`, and each CLI's resume arguments and output parsing
stay in its `agents/<agent_name>/launch_agent.py`. Other agents reject
`budget_mode: time` when the run starts; GEAK, Forge, and DeepSeek Harness use
their own budget settings.

## Agent cost records

For every task whose agent was launched, `main.py` writes
`<run_dir>/agent_costs/<task_workspace>.json`, for all agents. The record is
written after the agent returns or fails, and a cost-record error never changes
the task outcome. Each launcher only exposes what its provider reports; the
shared tool `src/tools/agent_cost.py` normalizes it and writes `null` when a
value is not reported. Arena never estimates dollars from tokens.

| Agent | Dollars | Tokens |
| --- | --- | --- |
| `claude_code` | `result.total_cost_usd` | `result.usage` |
| `codex` | `null` (not reported by the CLI) | `turn.completed.usage` |
| `cursor` | `null` (not reported by the CLI) | `result.usage`, when the CLI version emits it |
| `geak` | Claude Agent SDK `ResultMessage.total_cost_usd` | `ResultMessage.usage` |
| `forge` | KernelForge `llm_usage.total_cost_usd`, when `cost_available` is not false | KernelForge `llm_usage` counters |
| `deepseek_harness` | `null` | The `--json` final event's `usage`, when emitted; otherwise `null` |
| `task_validator` | `null` | `null` |

Main fields:

- `total_cost_usd` and `usage`: totals over every attempt and session, or `null`.
- `cost_status` and `usage_status`: `complete`; `partial` when the total is a
  lower bound (for example, a session stopped at its deadline reported nothing,
  or GEAK work finished after its last SDK result); `not_reported`; or
  `ambiguous` when resumed values cannot be combined safely.
- `attempts`: one entry per launch of the task, including a later
  `--resume-run` attempt, with run settings, agent seconds, any launcher error,
  and each raw provider report (`invocations`).

Resumed sessions are not double counted. Codex token usage and Claude Code
2.1.277 or later dollar totals are running totals for the session, so only the
increase is added. Older Claude Code versions report per process, and those
values are summed. If the Claude Code version cannot be parsed and a session
was resumed, the dollar total is `null` with status `ambiguous`. Token field
names are converted to snake_case but otherwise kept as the provider defines
them. For example, Codex `input_tokens` includes `cached_input_tokens`, while
Claude `input_tokens` excludes cache reads.

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
python3 src/tools/compare_runs.py \
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
