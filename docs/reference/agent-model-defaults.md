# CLI agent defaults and verification

Verified on **2026-09-15**, using the installed Linux x86-64 CLIs and the
authenticated accounts available to the maintainer. This is a dated verification
record, not a promise that every account or provider has the same model access.
The executable defaults remain in each agent's `agent_config.yaml`.

## Defaults and model evidence

| Integration | Selected model | Effort | CLI tested | Evidence |
| --- | --- | --- | --- | --- |
| [Codex](../../agents/codex/agent_config.yaml) | `gpt-6-astra` | `xhigh` | `codex-cli 0.154.0` | Official model documentation, local app-server `model/list`, successful inference and file-write smoke |
| [Claude Code](../../agents/claude_code/agent_config.yaml) | `claude-fable-5-1` | `max` | `2.1.272` | Official model configuration documentation, inference reporting this exact model, successful file-write smoke |
| [Cursor](../../agents/cursor/agent_config.yaml) | `composer-2.5` | Model-defined | `2026.09.10-fd3934a` | Account's `cursor-agent models` listing, inference reporting Composer 2.5, successful file-write smoke |

Codex moves from `gpt-5.5` to `gpt-6-astra` while preserving `xhigh`. The installed
CLI's model discovery returned `low`, `medium`, `high`, `xhigh`, `max`, and
`ultra` for Astra; `xhigh` is no longer the maximum. OpenAI's migration guidance
supports retaining the existing effort. This integration uses the CLI's
`model_reasoning_effort` setting, not a nonexistent `--effort` flag.
[Models](https://learn.chatgpt.com/docs/models),
[Astra migration guidance](https://developers.openai.com/api/docs/guides/latest-model/gpt-6-astra.md#migration-quickstart).

Claude moves from Sonnet 5 to Fable 5.1 while preserving `max`. Both
`claude-opus-5` and `claude-fable-5-1` succeeded in live probes; Fable 5.1 is the
newer, more capable option documented at verification time, so the user-supplied
"Opus 5" hint was not treated as an immutable model requirement. Fable 5.1 needs
Claude Code 2.1.257 or later. This is a model-tier change and can materially
increase cost; use a run-level override and, if desired, `max_budget_usd` for
budgeted experiments. It must not silently fall back to another model if access
is unavailable. Provider-managed content fallback can still occur; inspect the
returned `modelUsage` and transcript for the actual models used.
[Claude model configuration](https://code.claude.com/docs/en/model-config).

Cursor keeps its already-current Composer default. The account model list also
included `claude-opus-5-thinking-high`, `claude-sonnet-5-thinking-high`, and
`gpt-5.6-sol-high`. The documentation's model marketing names are not necessarily
CLI IDs. Use the installed CLI's account-specific listing to select IDs; do not
assume the same OpenAI/Anthropic ID is accepted by Cursor.
[Cursor CLI](https://cursor.com/cli),
[CLI parameters](https://cursor.com/docs/cli/reference/parameters).

## Run-level overrides

The three launchers now read known settings from the run config's `agent`
mapping, with precedence over their agent-local defaults:

```yaml
agent:
  template: codex
  model: gpt-5.6-terra
  effort: medium
  timeout_seconds: 3600
  max_iterations: 3
```

`model`, `timeout_seconds`, `max_iterations`, and `python_path` apply to all
three integrations. `effort` applies to Codex and Claude. Claude additionally
accepts an optional positive finite `max_budget_usd`, forwarded to its print-mode
CLI. This limit is not a run-wide cap across multiple tasks. `max_iterations` is
prompt guidance, not a CLI-enforced turn limit. An explicit `model: null` omits
the model argument and delegates selection to the CLI; omitting the field uses
Arena's checked-in default. Explicit `effort: null` similarly omits that flag.

```yaml
agent:
  template: claude_code
  model: claude-opus-5
  effort: high
  max_budget_usd: 5
```

Cursor has no standalone effort flag. Select an effort variant returned by
the account's model list:

```yaml
agent:
  template: cursor
  model: claude-opus-5-thinking-medium
```

An `agent.effort` setting for Cursor raises a configuration error instead of
being silently ignored. The installed CLI help also advertises bracket syntax
such as `claude-opus-4-8[context=1m,effort=high,fast=false]`. Arena forwards such
IDs literally, but their acceptance depends on the CLI/backend model catalog:
the live probe `claude-opus-5[context=1m,effort=medium,fast=false]` was rejected
with exit 1 and `Cannot use this model`. Do not construct IDs from marketing
names or assume that the advertised bracket feature works for every model.
`composer-2.5` does not acquire an effort knob merely because other models
support one.

These are **run settings**. Tasks must not name a particular agent, provider,
model, or authentication mechanism. Task schema and evaluation commands remain
owned by the shared framework and the
[task contract](../how-to/add-task.md).

## Invocation and failure contract

The launchers use these invocation forms. Codex and Claude read the prompt from
stdin; Cursor receives a literal prompt argument:

```text
codex exec --json --dangerously-bypass-approvals-and-sandbox
  --skip-git-repo-check --ephemeral -c features.memories=false
  --cd <workspace> --model <model>
  -c 'model_reasoning_effort="<effort>"' -- -

claude --print --verbose --output-format stream-json
  --include-partial-messages --permission-mode bypassPermissions
  --no-session-persistence --model <model> --effort <effort>
  [--max-budget-usd <budget>] --input-format text

cursor-agent --force --print --output-format stream-json
  --stream-partial-output --trust --workspace <workspace>
  --model <model> -- <prompt>
```

These are displayed across multiple lines for readability; launchers construct
argv lists, not shell command strings. Each resolved executable is invoked
directly, including paths with spaces. For stdin, the launcher passes a seekable
anonymous file containing the complete prompt. This avoids command-line size
limits and pipe-write stalls before timeout supervision starts. The same
transport is used by task_validator. Subprocess tests cover large Unicode
prompts; the historical live probes below predate this transport change.
Claude receives `IS_SANDBOX=1` and
`CLAUDE_CODE_DISABLE_AUTO_MEMORY=1` through its subprocess environment. Existing
permissive tool execution is retained for Arena's controlled runtime; it does
not turn a privileged container into a security sandbox. Cursor's `--trust`
applies to the prepared task workspace to avoid a headless trust prompt.
[Claude CLI reference](https://code.claude.com/docs/en/cli-reference),
[Cursor headless execution](https://cursor.com/docs/cli/headless).

All three retain the framework Python environment, report the CLI version and
resolved model settings, and capture streaming output. The launch command log
omits the prompt body and never dumps authentication files or environment values.
Codex and Claude disable persistent session recording for these one-shot runs;
Arena's own logs and candidate files remain available. Persistent learned-memory
disabling already present for Codex and Claude is preserved; these options do
not disable every user-configured plugin, MCP server, or instruction source.

A nonzero CLI exit, terminal failure event, or timeout raises an error rather
than returning as successful agent execution. Timeouts and interrupted waits
terminate the invocation's process group, including tool/compiler children.
Candidate files are retained. The caller must record the agent failure and
decide whether/how to evaluate a retained candidate; a successful CLI process
alone never establishes kernel correctness or performance.

## Verification performed

No CLI installation, login state, task source, or GPU job was changed for these
checks. Version/help inspection used:

```bash
codex --version
codex exec --help
claude --version
claude --help
cursor-agent --version
cursor-agent --help
cursor-agent models
```

Codex discovery used the installed `codex app-server --stdio`: after
`initialize`/`initialized`, request `model/list` with `includeHidden: false`.
Only model IDs and supported effort metadata were retained. It returned
`gpt-6-astra` as the default and also listed `gpt-5.6-sol`, `gpt-5.6-terra`, and
`gpt-5.6-luna`.
[App-server protocol](https://learn.chatgpt.com/docs/app-server).

Live probes used independent temporary directories, an instruction to return
`ARENA_CLI_OK` without tools, and a 90- or 120-second external timeout. Codex used
read-only mode, `--ignore-user-config`, and `--ephemeral`. Claude used print mode,
`--tools ''`, `--strict-mcp-config`, `--setting-sources ''`, disabled hooks,
`--no-session-persistence`, and a USD 1 per-request CLI budget. Cursor used ask
mode. Only selected result/model/usage fields were retained, not raw auth or
account details.

| Probe | Observation |
| --- | --- |
| Codex `gpt-6-astra`, `xhigh` | Exit 0, exact token, 14.16 s; `turn.completed` reported 9 output tokens |
| Codex `gpt-5.6-terra`, `medium` | Exit 0, exact token, 13.36 s; medium-effort option for subsequent validator runs |
| Claude `claude-opus-5`, `max` | Exit 0, exact token; assistant and usage named `claude-opus-5`; 2.23 s |
| Claude `claude-fable-5-1`, `max` | Exit 0, exact token; assistant and usage named `claude-fable-5-1`; 2.67 s |
| Cursor `composer-2.5` | Exit 0, exact token; init model `Composer 2.5`; 6.87 s |
| Cursor `claude-opus-5-thinking-medium` | Exit 0, exact token; init model `Claude Opus 5 300K Medium`; 6.98 s |
| Cursor parameterized Opus 5 ID above | Exit 1, `Cannot use this model`, 1.60 s; no inference success claimed |

The Cursor listing labeled the Opus variant as 1M Thinking, while the live init
event used the 300K Medium label above. Neither the short smoke nor the model
name establishes an effective context-window guarantee; retain actual runtime
metadata when comparing agents.

Claude's probe summaries reported list-price estimates of USD 0.029304 for Opus
and USD 0.074814 for Fable, including small Haiku utility calls. These are not
subscription invoices or estimates for GPU optimization workloads.

The updated **real Arena launch functions** were then exercised with their
default models/efforts and a minimal prompt builder replacing only task prompt
construction. Each agent had to create `smoke_result.txt` containing exactly
`ARENA_ADAPTER_OK\n`. All three files were independently checked; the launchers
also returned the token. Codex completed in 8.59 s, Claude in 7.40 s, and Cursor
in 10.73 s. Each had `timeout_seconds: 120` and `max_iterations: null`; Claude
also had `max_budget_usd: 1`. This checks actual argv, authentication, streaming,
workspace selection, and a file-writing tool call, not the task evaluator or
GPU kernels.

CPU regression command:

```bash
python3 -m pytest -q tests/test_cli_agents.py
```

The focused suite passed **68 tests**. It covers model/effort overrides,
literal argv and paths with
spaces, malformed settings, optional spending limits, explicit CLI errors,
failure events with exit code zero, and timeout cleanup of an actual child
process. It uses fake CLIs and requires no credentials or GPU. The test
environment was CPython 3.12.3, pytest 9.1.1, and PyYAML 6.0.3 in an isolated
temporary virtual environment. Live checks are not run by pytest.

Full Docker/GPU optimization, schema-v2 task prompting, task-validator coverage,
and the requested multi-agent task matrix belong to the integration validation.
This record does not claim those checks passed or establish model-quality
rankings from short smoke requests.
