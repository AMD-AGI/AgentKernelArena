# GEAK for Arena schema v2

`agents.geak.launch_agent.launch_agent` runs the real upstream GEAK dispatcher
and single-language lane through Claude Code's dynamic `Workflow` tool. GEAK's
TechLead plans, author engineer creates seeds, specialist engineers search,
verifiers check patches, and the Director validates. No Codex substitute or
task-family dispatcher is involved.

The adapter accepts `ARENA_TASK_CONTEXT` version 1 from an accepted `TaskSession`.
It loads the shared `TaskSpec`, captured case manifest, candidate workspace,
and independent baseline workspace. It supports declared `hip`, `triton`, and
`flydsl` languages; implemented and unimplemented candidates; provided and
frozen initial-candidate baselines; nested/multiple files; file, tree, and
Python symbol scopes. Initial-language translations use GEAK's author mode.
Backend availability and actual generated-code correctness are decided by the
task's public runner and the selected GPU runtime, not the file extension.

## Parent integration contract

The parent integration must add `GEAK = "geak"` to `AgentType`, map its launcher
to `agents.geak.launch_agent.launch_agent`, and select ordinary optimization
post-processing. The launcher builds its context-based contract internally and
does not call an agent-specific shared prompt builder. Retain the standard v2
orchestration path that supplies `ARENA_TASK_CONTEXT` and independently calls
the evaluator after the launcher. This worker intentionally does not modify
`src/module_registration.py`, `main.py`, shared task APIs, or tasks.

Existing `geak_v3`, `geak_v3_triton`, and `geak_v4` launchers now delegate schema
v2 to this same adapter before probing their legacy CLIs. Their v1 behavior is
unchanged. Keep their existing registry entries if backward compatibility is
wanted; no additional v2 backend-specific registry identifiers are necessary.
V2 aliases use `agents/geak/agent_config.yaml` and its run-level overrides.

## Runtime setup

Provide `GEAK_HOME` pointing to a clean upstream checkout at the revision pinned
in [compatibility.py](compatibility.py). The legacy `GEAK_V4_WORKFLOW_DIR` can
instead point to its `kernel_workflow` directory. The pin checks git revision,
clean engine/knowledge trees and the dispatcher/lane SHA-256 digests; another
revision fails preflight and requires an explicit compatibility update.

Install [requirements.txt](requirements.txt) into the agent's runtime Python.
Arena and PyYAML must also be importable. `GEAK_PYTHON` selects that interpreter;
otherwise the launcher uses its own Python. `GEAK_CLAUDE_BIN` selects Claude Code
or the adapter resolves `claude` on PATH. A Claude build with the dynamic
Workflow tool and non-root Docker execution are required by the existing SDK
lifecycle runner. Authentication is inherited from the runtime environment.
Use Arena's `AGENT_KERNEL_ARENA_PYTHON` for public task commands when their
GPU/toolchain Python differs from the SDK interpreter.

Configure `agent.template: geak` after the parent registry integration.
Run-level `agent.model`, `effort`, `budget`, `deep_cost`, `min_improve`, and
`timeout_seconds` override [agent_config.yaml](agent_config.yaml). A null model
uses the Claude runtime's configured default. Credentials are never agent
configuration fields. GPU IDs use the process-visible namespace, including
logical device zero in a single-GPU Arena worker.

GPU experiments continue through Docker:

```bash
make docker-check-agents CONFIG=<run-config>
make docker-run CONFIG=<run-config>
```

Generic registry/preflight selection is a parent integration responsibility;
this worker's CPU checks do not qualify those Docker commands or a GPU task.

## Upstream compatibility and evaluation

The pinned upstream engine assumes `kernel_src/`, `unittest.py`, `meta.json`,
and sometimes a vendor-specific frozen baseline. Those assumptions do not
implement the Arena task contract. Each invocation therefore creates its own
engine/knowledge copy and applies these explicit adaptations:

- Supply deterministic setup and benchmark inputs from Arena's context and
  freshly executed baseline compile/performance actions.
- Append the Arena contract to every GEAK agent call, including inline calls,
  and replace author/Director/profile role instructions in the private copy.
- Route compile, correctness and performance through [bridge.py](bridge.py),
  which calls `run_action` with the task's declared argv commands. Correctness
  rebuilds; performance rebuilds and checks first. The captured manifest and
  baseline/candidate timing-method pairing are enforced.
- Replace the private workspace copier with a generic, fresh-destination copy
  preserving task-relative inputs and private git history. Disable artifact
  reclamation, external knowledge warm starts, learned writes and web research.
- Stop new rounds on a failed clock query. One invocation deadline covers
  setup, baseline actions, SDK execution, lock waits, checks and delivery.
  Filesystem copying checks that deadline between operations. The supervised
  SDK process and its descendants are killed on timeout, including runner
  descendants that started separate sessions.

GEAK engineers share a deadline-bounded file lock for device actions, avoiding
overlapping measurements within the run. No task-owned Python module is
imported into the agent, no framework helper is injected into a task, and no
baseline is reconstructed from a generated candidate. The framework's accepted
initial context owns baseline correctness/diagnostic policy; the bridge does
not relabel diagnostic failures as ordinary passes.

## Delivery and failure semantics

Run artifacts are in a unique hidden sibling of the Arena workspace. They
include `context.json`, `job.json`, `engine_identity.json`, per-action structured
checks, GEAK's private workspaces, `engine_result.json`, and `delivery.json`.
The launcher logs the run directory and retains it on failure. Its own artifacts
omit runtime environment dumps, raw task stdout and model transcripts.

After GEAK returns, the launcher independently compiles, checks and times the
complete canonical candidate, then installs only declared editable files with
their relative directories. This includes the initial author seed and newly
added helpers, which a patch relative to the seed's commit could omit. A correct
candidate is deliverable even below 1x speedup. Protected files and baseline
sources are checked before/after public actions; runner source mutation,
missing cases, changed identities/methods, symlink escapes and nonzero exits fail.

`delivery.json` distinguishes engine completion, attempted/failed/delivered
files, and retained workspace source hashes. Engine failure remains a launcher
failure even if a checked candidate was delivered. Timeout leaves private
artifacts for diagnosis; it does not invent successful validation. An I/O error
during delivery remains a failure and records the actual retained sources.
The adapter never writes `task_result.yaml` or a validator report. Arena's
independent evaluator and framework exports remain authoritative.

## Security and reproducibility review

New execution paths are the selected Python SDK worker, Claude's real Workflow
runtime, private Git initialization, and the task-declared public argv commands.
The adapter does not install/download dependencies at run time or modify the
shared GEAK checkout. Its subprocess environment supplies authentication without
embedding credentials in prompts, argv, job configuration, or adapter logs.
The copied engine is pinned and its adapted lane digest is recorded. Upstream
role defaults that create harnesses or publish shared knowledge are overridden
inside the per-run copy. These are reproducibility boundaries within Arena's
existing permissive container model, not a security sandbox.

## CPU validation and remaining qualification

The focused suite uses executable CPU fixture runners with synthetic protocol
timings. With a configured upstream checkout and Node it also executes the
actual dispatcher/lane, stubbing only model/runtime agent calls. It exercises
authoring, planning, engineers, verification, commit and Director phases for
all three language declarations. This proves interfaces and control flow,
not generated HIP/Triton/FlyDSL compilation, numerical GPU correctness, or
measured speedups.

```bash
GEAK_TEST_CHECKOUT="$GEAK_HOME" GEAK_TEST_NODE=<node-binary> \
  python3 -m pytest -q tests/test_geak_schema_v2.py tests/test_geak_v4.py
```

Optional probe dependencies can live in the ignored `agents/geak/.scratch-env`
environment; do not install into a shared GEAK or global environment. SDK
interface imports were checked with the version in `requirements.txt`; the
JavaScript probe used Node 24.19.0. A live authenticated Workflow invocation,
Docker qualification, large image-task copy cost, and actual GPU campaigns
remain untested here. Generic profiling is explicitly unavailable because the
v2 public protocol has no profiling action; no evaluator-tool analysis is
claimed. GEAK bakeoff, cross-run resume and external knowledge publication are
not exposed by this adapter.
