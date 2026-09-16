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

## Arena integration

Select `agent.template: geak`. The shared registry loads this launcher and uses
ordinary optimization post-processing. The launcher builds its contract from
`ARENA_TASK_CONTEXT`; the standard v2 pipeline independently evaluates the
retained candidate after GEAK returns. Task files do not contain GEAK-specific
drivers or engine layout requirements.

The retained `geak_v4` launcher delegates schema v2 to this adapter before
probing its legacy CLI. Its v1 behavior is unchanged. The `geak_v3` and
`geak_v3_triton` names are retired; select `geak` for their v2 tasks.
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

Configure `agent.template: geak`.
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

Docker provisions Claude for `geak` and its v2 aliases, mounts only the selected
GEAK checkout read-only, and forwards `GEAK_HOME`. The complete checkout is
needed for revision checks and private engine/knowledge copies. Preflight
installs the pinned SDK into the GEAK-only dependency directory when necessary
and verifies the clean upstream pin. These checks do not certify a live
Workflow invocation or a GPU task.

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

Failed public actions retain bounded, redacted task reasons in `checks/` and
return those diagnostics to GEAK. `runtime_identity.json` separately records
known provider error codes and available rate-limit metadata, without raw
provider messages. A rate-limit event with `status: rejected` indicates a
rejected request; `overage_status: rejected` alone does not establish that the
regular allowance is exhausted.

The private, atomically written `runtime_identity.json` also retains bounded
`sdk_diagnostics`: observed Workflow call/match counts, trusted field names and
type differences (no argument values), nested exception classes with fixed
reason codes and actual exit codes when available, and allowlisted CLI stderr
codes. Counts cover only the observed SDK message prefix; they do not recover
an earlier CLI session. Unknown extra keys are counted without copying their
names; the known `run_in_background` key can retain its type. Raw stderr,
exception text and tracebacks are not retained. These fields
are diagnostic evidence only and do not change native completion or acceptance.
On Python 3.10, grouped failures use the `exceptiongroup` backport when present;
otherwise plain failure diagnostics remain available. Group support is resolved
only when recording failures, so dry-run needs neither the SDK nor its backport.

Read these diagnostics alongside `engine_result.json`, `delivery.json`, and
Arena's `task_result.yaml`. Candidate acceptance means the retained source
passed Arena's evaluation. It does not establish that GEAK completed its
workflow: an interrupted or failed engine can still leave an accepted candidate.
Count a completed optimization only with evidence of native GEAK search and
successful workflow completion as well as independent candidate acceptance.
The v2 launcher requires the invoked Workflow tool's own return, correlated by
tool identity with its synchronous result or completed background notification.
The invocation must match the prepared script and arguments. Assistant prose,
Director markers, and an assistant-written `workflow_return.json` cannot prove
native completion. A failed or missing runtime return remains an engine failure.
The dispatch prompt supplies one exact JSON object containing only `scriptPath`
and `args`, explicitly excluding extra keys such as `run_in_background`. This
clarifies the requested call without guaranteeing model compliance or forcing
synchronous execution; matching native background completion remains supported.
The outer `args` omit the two duplicate full-contract fields. A hash-checked
private copy of the native dispatcher restores `arena_contract` and `task` from
trusted JSON string literals before the author/optimize lane receives its
arguments. The lane and role inputs retain the full contract; model output
budgets stay unchanged. Adapter identity version 3 records the adapted
dispatcher hash as well as the adapted lane hash. Its dispatcher accepts either
an object or one JSON-encoded object string and explicitly decodes the latter
with `JSON.parse` before restoring the trusted contract. It rejects null,
arrays, scalars, duplicate keys at any depth and non-finite numbers.

The prepared engine's `args_transport` handoff opts into matching that exact
version-3 dispatcher SHA-256. The SDK checks that pin and the decoder before
launch; generic workflow-runner callers remain strict unless they supply the
validated opt-in. Only `scriptPath` and `args` are accepted at the outer tool
boundary. The complete decoded arguments must match, including every nested
key and value. Numeric comparison preserves exact Python integer/float equality;
booleans and strings never coerce to numbers. V3 rejects integer-valued numbers
outside JavaScript's safe range `[-(2**53 - 1), 2**53 - 1]` in both Python and
the actual dispatcher, before accepting or forwarding arguments. Default strict
callers retain large-integer distinctions instead of rounding them to floats.
Native tool-ID/return correlation, invocation counts,
timeouts and model budgets are unchanged.

Raw SDK tool inputs are never rewritten. Diagnostics distinguish the raw
`args_encoding`, a SHA-256 of its JSON serialization (`ensure_ascii=True`,
`sort_keys=True`, Python's default separators), the comparison mode and the
normalized root type. Argument values still stay out of the diagnostic report;
original native capture artifacts retain their own raw representation. External
evidence collectors can use `workflow_inputs_match` from
[argument_transport.py](argument_transport.py), passing the prepared engine's
`args_transport`. They must bind that handoff to `engine_identity.json` version
3 and its dispatcher hash; string input alone never authorizes normalization.

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

## Validation evidence and remaining scope

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
JavaScript probe used Node 24.19.0. These CPU checks are separate from the actual
authenticated Workflow and Docker/GPU campaign evidence in the
[dated verification checkpoint](../../docs/reference/verification-2026-09-15.md#agent-matrix-checkpoint).
That checkpoint records five historical completed GEAK tasks with accepted
candidates, retaining their original source/model identities. The source-39
records lack full tool-ID/session correlation and do not qualify the later
strict-return gate.
A subsequent real CPU Workflow probe verified matched native completion on
`27f28461`; it adds no GPU or matrix credit. Native author activity in the
continuation is also partial evidence, not proof of a completed planner,
engineer, verifier and Director sequence with a matching strict Workflow return.
Neither these probes nor author activity establish completion of the 180-pair
matrix or quantify large image-task copy cost.

Generic profiling is explicitly unavailable because the
v2 public protocol has no profiling action; no evaluator-tool analysis is
claimed. GEAK bakeoff, cross-run resume and external knowledge publication are
not exposed by this adapter.
