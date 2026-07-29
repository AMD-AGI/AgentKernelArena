# GEAK v4 Agent

The `geak_v4` integration runs GEAK's deterministic
`kernel_workflow/kernel_workflow.js` through Claude Code's dynamic **Workflow**
tool. GEAK optimizes the kernel and, when its Director validation passes, applies
the validated patch directly into the task workspace (`apply_to_original="true"`).

**AgentKernelArena is the single source of truth for scoring.** After GEAK
finishes, AKA verifies its harness is intact, re-materializes the perf helpers,
and independently re-evaluates the kernel (compile → correctness → performance);
GEAK's own numbers are not used for the final result.

Like the `forge` and `claude_code` agents, `geak_v4` relies purely on the
environment — it touches nothing outside `agents/geak_v4/`. The launcher stays
thin: it writes a versioned handoff and shells out to `workflow_runner.py`.

## Why a runner instead of a plain `claude -p`

On current Claude builds the `Workflow` tool runs as a **background task**: the
main agent turn ends immediately, so a one-shot `claude -p` would return (and
tear the still-running workflow down) before GEAK finishes. `workflow_runner.py`
keeps a persistent `claude_agent_sdk` client alive and drives completion off the
SDK's background-task lifecycle (`TaskStartedMessage` → `TaskNotificationMessage`)
plus GEAK's on-disk terminal marker. It is a kernel-scoped analogue of GEAK's own
`interface/run_e2e.py` and mirrors its `handoff.json` / `result.json` contract.

## Prerequisites

- An AMD Instinct GPU and a supported profiler (`rocprof-compute`).
- A local GEAK checkout. Point `GEAK_V4_WORKFLOW_DIR` at its
  `kernel_workflow/` directory (default: `/opt/geak/kernel_workflow`).
- Claude Code 2.1.177 or newer, installed and logged in (the minimum version for
  the dynamic Workflow feature), plus access to the model configured in
  `agents/geak_v4/agent_config.yaml`.
- The `claude-agent-sdk` Python package installed in the interpreter that runs
  the agent.
- For a `flydsl2flydsl` task, FlyDSL available in the environment.

## Setup

```bash
# 1. Clone GEAK and expose its kernel_workflow directory.
git clone https://github.com/AMD-AGI/GEAK.git
export GEAK_V4_WORKFLOW_DIR=/absolute/path/to/GEAK/kernel_workflow

# 2. Authenticate Claude Code and confirm the Workflow-capable version.
claude --version
claude            # log in
claude auth status

# 3. Install the SDK the runner uses to hold the background Workflow open.
pip install claude-agent-sdk
```

This adapter was developed against GEAK commit
`4965d5b2ccde927925c8c5501a25c1233daa52eb` (`v4.0.0-102-g4965d5b`). For a
reproducible review, check out that revision; newer GEAK revisions may require an
adapter/schema update.

## Run the example

The included MI300/MI300X example runs one HIP GELU task. Run it in an
environment that already satisfies the prerequisites above (`GEAK_V4_WORKFLOW_DIR`
set, `claude-agent-sdk` installed, `claude` logged in):

```bash
python main.py --config_name example_configs/quickstart_geak_v4_mi300.yaml
```

## Supported task types

- `hip2hip`
- `triton2triton`
- `flydsl2flydsl`

These are single standalone kernels. The launcher reads `source_file_path` to
steer the optimizer ("optimize only these files") and fails early if the declared
anchor source is missing. Authoring, translation, repository, and image-level
tasks are out of scope for this integration.

## Artifacts

For a task workspace named `<workspace>`, GEAK's run artifacts live OUTSIDE the
scored workspace, under a hidden sibling directory:

```text
.<workspace>_geak_v4/<run-id>/
├── eval/           # GEAK validation + final_patch.diff
├── runs/           # GEAK experiment tree (its own copy of the kernel)
├── handoff.json
└── result.json
```

They sit beside the workspace (not inside it) because GEAK copies `kernel_path`
into its own experiment tree — nesting outputs under the workspace would recurse,
and it keeps GEAK's scratch clear of the directory Arena scores.

## Integrity boundary

`geak_v4` lets GEAK edit the workspace in place, exactly like `forge` and
`claude_code`. Integrity is enforced by the Arena harness, not the launcher:
`main.py` snapshots the harness before the run, verifies it afterwards,
re-materializes the perf helpers, and re-scores the kernel independently. A run
that tampers with a protected harness path fails harness verification. This is a
fail-closed correctness control, not an OS security sandbox; a failed run is not
automatically rolled back. A real paid Claude workflow invocation is not part of
the offline tests — run the one-task example with an authorized account before
relying on this integration for a benchmark campaign.
