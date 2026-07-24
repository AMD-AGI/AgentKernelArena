# GEAK v4 Agent

The `geak_v4` integration runs GEAK's deterministic
`kernel_workflow/kernel_workflow.js` through Claude Code's dynamic Workflow
tool. GEAK works on a disposable copy of the task workspace; AgentKernelArena
imports only a validated patch for the one declared kernel source and then runs
its normal correctness and performance evaluation.

## Prerequisites

- An AMD Instinct GPU and `rocprof-compute`, as required by the Arena Docker
  smoke/run contract.
- A local GEAK checkout. By default, the Docker runner looks for `GEAK` beside
  the Arena checkout:

  ```text
  parent/
  ├── AgentKernelArena/
  └── GEAK/
  ```

- Claude Code 2.1.177 or newer, installed and logged in on the host. The minimum
  version is required for the dynamic Workflow feature.
- Access to the Claude model configured in
  `agents/geak_v4/agent_config.yaml`.
- FlyDSL installed with `make docker-setup-flydsl` when selecting a
  `flydsl2flydsl` task.

## Setup

Clone GEAK beside AgentKernelArena, authenticate Claude Code on the host, and
install the Python SDK into the persistent container dependency directory:

```bash
cd /path/to/parent
git clone https://github.com/AMD-AGI/GEAK.git
cd AgentKernelArena

claude --version
claude
claude auth status

make docker-setup-geak
```

This adapter was developed against GEAK commit
`4965d5b2ccde927925c8c5501a25c1233daa52eb`
(`v4.0.0-102-g4965d5b`). For a reproducible review, check out that revision;
newer GEAK revisions may require an adapter/schema update.

If the GEAK checkout is elsewhere, provide its absolute host path for setup,
preflight, and every run:

```bash
export AKA_GEAK_ROOT=/absolute/path/to/GEAK
make docker-setup-geak
```

`make docker-setup-geak` installs only `claude-agent-sdk`. It does **not**
`pip install` GEAK. The Docker runner bind-mounts the checkout read-only at
`/opt/geak`, so the workflow code, roles, and knowledge remain unchanged by an
Arena run. The persistent Python dependency directory is writable only in the
explicit setup container and is nested-mounted read-only during agent runs.

## Run the example

The included MI300/MI300X example runs one HIP GELU task:

```bash
CONFIG_PATH=example_configs/quickstart_geak_v4_mi300.yaml
make docker-check-agents CONFIG="$CONFIG_PATH"
make docker-run CONFIG="$CONFIG_PATH"
```

`docker-check-agents` verifies the Claude login and version, the Agent SDK, the
GEAK workflow checkout, and an available profiler without starting an
optimization.

## V1 supported task contract

V1 of the Arena integration supports:

- `hip2hip`
- `triton2triton`
- `flydsl2flydsl`

A task must declare exactly one existing source in `source_file_path`. That
source must be a normal, non-symlink file and cannot be a protected config,
test, or harness path. In particular, names such as `test_*.py`, `*_test.py`,
`test_*.cpp`, `*_test.hip`, and `*_harness.cu` are rejected. Tasks with
multiple sources or a source that also contains the test harness are
intentionally out of scope.

Authoring, translation, repository, and image-level tasks are not supported by
V1. Use a supported task with a single standalone kernel source.

## Isolation and artifacts

For a task workspace named `<workspace>`, GEAK artifacts are kept outside the
scored workspace under:

```text
.<workspace>_geak_v4/<run-id>/
├── input/          # Disposable task copy seen by GEAK
├── eval/           # GEAK validation and final patch artifacts
├── runs/           # Workflow experiment artifacts
├── handoff.json
└── result.json
```

The hidden artifact directory is a sibling of the task workspace. GEAK is
instructed not to apply changes to the original input, and the Arena accepts a
result only when GEAK's Director validation passes and the final patch changes
the one allowlisted source without creating, deleting, renaming, or changing
the mode of a file. Arena also compares a full manifest of the scored workspace
before and after Workflow execution; a direct mutation makes the task fail
before scoring or patch import.

## Validation boundary

Offline tests cover handoff mapping, result parsing, patch filtering, artifact
isolation, and Docker argument construction without contacting Claude. A real
paid Claude workflow invocation is not part of that offline validation; run
the one-task example with an authorized account before relying on this
integration for a benchmark campaign. As with other Arena task workspaces, the
disposable-copy and manifest checks are fail-closed integrity controls, not an
OS security sandbox; a failed run is not automatically rolled back.
