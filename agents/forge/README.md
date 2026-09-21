# Forge agent

Forge is one agent integration for the [public task v2 contract](../../docs/how-to/add-task.md).
The task declares its candidate, baseline, entrypoints and evaluation commands.
It does not supply a Forge driver or consume `KERNELFORGE_*` variables. The registry maps the old
`forge_operator2flydsl` name to this same launcher and configuration.

## Workflow selection

| Current candidate | Internal workflow |
| --- | --- |
| Verified target implementation | `forge-loop` |
| Unimplemented FlyDSL candidate | `forge-rewrite-by-flydsl`: PORT, then its nested loop |
| Existing implementation in a different language, target FlyDSL | Same rewrite workflow |
| Unimplemented or cross-language HIP/Triton candidate | Forge correctness-only implementer, then `forge-loop` |
| Initialization to another target language | Explicit unsupported-capability error until that backend is qualified |

`workflow: auto` is the default. `optimize` requires a currently verified target
implementation; `rewrite` requires initialization. A changed candidate is rebuilt
and checked against the final target contract before routing, so a resumed task
whose config originally declared a stub does not repeatedly PORT. A changed
candidate that compiles but receives a complete, explicitly classified numerical
rejection enters initialization again. Its failed evidence is retained and fed
to the implementer; it is never treated as an accepted implementation. Missing
failure classifications, build failures, runtime errors, incomplete case evidence
and unchanged broken task seeds still abort. This deliberately narrow recovery
does not infer a candidate mistake from a missing dependency or broken harness.
The backend comes from `candidate.language` and is checked against the actual
installed KernelForge registry. There is no task-name dispatch, suffix inference,
or silent substitution of an unsupported backend with FlyDSL.

The FlyDSL CLI still owns its PORT and nested OPTIMIZE. HIP/Triton use the same
KernelForge implementer factory and provider execution mechanism as the loop,
with the upstream `correctness_only` gate enabled during initialization. This
phase is implemented in [initialization.py](initialization.py); it does not call
another Arena agent or require a task-specific driver, category, factory name,
or initialization command. `rewrite` explicitly requests initialization and
selects the appropriate path for the declared target language.

`initialization_budget_fraction` bounds HIP/Triton initialization against the
remaining shared campaign time. It reserves the remainder for native search,
whose own initial measurements, analysis and round admission also take time.
FlyDSL PORT is bounded instead by `max_port_attempts` and the engine's own phase
budget. A correct PORT is still not evidence of a completed search round; the
native iteration records and final Arena acceptance establish that.

Native driver timeouts cover the task's complete action sequence: compile and
correctness for validation; compile, correctness and performance for candidate
benchmarking. The adapter derives these outer ceilings from the public task
timeouts instead of the upstream five-minute benchmark default. Every action
keeps its own declared timeout, and the shared campaign/initialization deadline
still applies. The implementer session budget also includes its in-session
checks, so a task with long checks needs enough session time as well as campaign
time; increasing a driver ceiling does not extend either budget.

Every public action keeps an append-only diagnostic record under the Forge
artifact root's `action-evidence/` directory, outside its temporary build tree.
Each unique action directory contains a start record and, when the action
returns or raises `TaskExecutionError`, its full argv, stdout, stderr, return
codes, elapsed times and protocol result or execution error. The records bind
the task, role, action, bridge invocation, public invocation ID when available,
context digest and declared input file hashes. The optional framework source
inventory is bound by path and digest; these diagnostics do not archive an
entire build or claim to hash every runtime dependency. They include no process
environment or authentication configuration. Build cleanup leaves them intact.
An interrupted start or partial JSON write is not a completed action record.

For both a protocol-reported failure and an execution error, the engine receives
a short error with the full evidence path, plus the exit code and last 6,000
characters of command output as a single JSON-escaped diagnostic line. The
nonzero exit keeps the engine from scoring that output. Diagnostic persistence
does not change task commands, checks, benchmark boundaries or score selection.

The independent baseline remains the score denominator. Whether a first
candidate slower than that baseline can establish the engine's incumbent is the
engine's own policy; the adapter no longer overrides it. A first valid
implementation is recorded as the first scoreable candidate rather than as a
speed improvement, and Arena scores the delivered bundle either way.

### HIP/Triton initialization

Initialization uses the existing Forge correctness-only implementer before
entering the ordinary performance search:

1. Keep the task's provided baseline and references independent. Measure baseline
   timing through the public baseline actions. Preserve existing candidate stubs
   or cross-language sources; create empty files only for missing declared file
   targets. An empty implementation tree needs a declared entrypoint anchor.
2. Run `kernelforge.orchestrator.agent.make_agent_fn` with the target backend's
   expertise prompt, normal configured provider/model, real SDK session execution,
   workspace guard and correctness-only in-session gate. The task instructions,
   evaluator and declared entrypoints supply the implementation contract.
3. After each session, independently run candidate compile and complete
   correctness through the same public runner. A provider's text or gate flag
   cannot substitute for that check. Feed concrete failures and partial code into
   the next Forge attempt. Reject harness changes, Git history changes and
   unreaped-workspace contention. Record session identity, failures and usage.
4. Commit the complete validated candidate bundle, including new declared tree
   helpers. Enter the real `forge-loop` once, passing the independent baseline
   timings and the refreshed file list. The loop measures the actual initialized
   candidate as its first incumbent.

The first valid implementation need not beat the production baseline. Its
acceptance says nothing about final Arena scoring. If a successfully completed
loop reports no best commit because no iteration earned KEEP, delivery can select
the recorded initial correct commit; its status says
`delivery_selection: initial_correct_implementation`. The loop's measurements and
iteration count remain unchanged. An engine error or timeout still fails the run.

The phase has a bounded attempt count and uses at most
`initialization_budget_fraction` of the remaining campaign budget, reserving the
rest for the loop. Each provider session also honors `session_timeout_seconds`.
That limit covers the whole implementer invocation, including provider resume
turns and their in-session checks. A timed-out session unwinds the provider and
records `session_timeout`; it does not claim a passed gate. The native loop can
then assess the remaining candidate through the complete public task checks and
save or reject it within the shared campaign deadline. Protected-file integrity
is finalized on cancellation as well as on a normal return.
The FlyDSL PORT loop may retry an expired attempt within its remaining phase
budget; exhausting the whole phase still stops PORT.
Failure to initialize within those bounds prevents loop launch and leaves the
original candidate unchanged. The adapter saves `initialization.json` outside the
editable engine tree, including failed attempts. Its usage is combined with the
loop's usage; the separate records remain available for auditing.

## Evaluation bridge

The framework must provide `ARENA_TASK_CONTEXT`, a protected JSON file outside
both task workspaces, containing:

```json
{
  "version": 1,
  "task_id": "suite/task",
  "task_config": {"schema_version": 2},
  "workspace": "<absolute runtime candidate root>",
  "baseline_workspace": "<absolute independent baseline root>",
  "manifest": {"protocol": "arena-eval-v1"}
}
```

The abbreviated `task_config` and `manifest` above stand for the complete
normalized TaskSpec mapping and successful `validate-task` ActionResult. These
are framework runtime paths, not paths authored in a task config. The adapter
validates them with `TaskSpec`, `parse_command_result` and `CaseManifest`.

Forge generates `arena_forge_driver.py` in its private engine workspace:

| KernelForge request | Public task actions |
| --- | --- |
| Default / `--mode` | Candidate compile, correctness |
| `--bench-mode` | Candidate compile, correctness, performance |
| `--ref-bench-mode` | Baseline compile, performance |
| `--profile-run` | Explicit unsupported response; no public profiling action exists |

Baseline numerical validation is the framework's initial-validation policy;
the driver never substitutes baseline correctness for candidate correctness.
Every invocation copies the protected task layout into a private assessment
workspace and overlays only the current candidate bundle. All actions within
that invocation use that same copy. Baseline assessments start from the separate
baseline snapshot. Task scripts are executed through `src.task_execution.run_action`;
no task Python modules are imported into the agent adapter.

Before creating a native agent, the adapter requires `config.workspace` to be
the engine/lane Git root and checks that the task config, generated driver and
existing declared protected files are tracked. A nested rewrite attempt or an
incomplete protection inventory fails before the agent starts.

Every generated phase prompt preserves the task's implementation and dependency
constraints and gives them priority over backend guides, examples, and knowledge
base suggestions. A passing driver does not waive those constraints. This prompt
guidance is not a dependency enforcement mechanism; task evaluation must still
check the implementations it accepts. A prohibited replacement operator remains
prohibited when that library operator happens to use the target language internally.

Task references, comparison rules, inputs, timings, and complete case manifest
remain authoritative. The driver translates the task verdict into `allclose`;
it does not invent an SNR threshold or alter tolerances. Forge stage hints such
as `--warmup` and `--iters` do not reduce the task's workload or sample counts.
Unknown case selectors are rejected. Case IDs are reversibly encoded for
KernelForge's whitespace-delimited output without merging distinct IDs.

Every performance row must have the expected identity and finite positive timing.
The adapter reports per-case times and an honestly named `mean_ms` summary.
It supplies the independent per-case baseline to both workflows. KernelForge
uses per-case baseline/candidate ratios for its search objective, alongside its
own KEEP policies; Arena still performs its own final evaluation and scoring.
No agent log or `port_ok` field becomes an Arena verdict.

## Candidate delivery and isolation

All workflows run in a fresh sibling directory of the task workspace. The
adapter preserves complete relative paths, dependent files, and declared
implementation trees. A rewrite binds its current attempt to the same declared
paths when assessing it; the task never needs to know the attempt's location.
The final bundle is read from the engine's selected full Git commit, not from a
newest-file guess or an unfinished working-tree edit. Missing files, candidate
symlinks, path escapes and malformed result bindings are errors. Symbol-scoped
Python files retain their protected statements and tests.

The status record separates `DELIVERED` from the eventual Arena verdict, which
remains `pending`. A correct candidate can be delivered without an improvement.
If a successful optimization search reports no KEEP and omits `best_commit`,
the adapter retains the exact starting candidate commit that passed its public
checks. This also covers a validated generated implementation on resume. The
status records `delivery_selection: initial_validated_implementation`; discarded
search files are not installed. A claimed later best without its commit, missing
iteration metadata, or an unsuccessful engine cannot use this fallback. Arena
still checks and measures the delivered bundle independently, and retaining it
does not claim an optimization gain or prove that a search iteration ran.
A failed PORT, unexplained nonzero engine exit, missing structured result, or a
timeout that recovered no candidate reports failure and leaves the original task
candidate unchanged. Diagnostic scratch, logs, result JSON and selected artifact
hashes are preserved in the fresh Forge artifact directory; old experiment
directories are never removed. The only disposable copy is the baseline tree,
which lives in a temporary directory removed when its measurement ends; the
candidate is measured in place, so a campaign makes no copy per assessment.
Each driver invocation reaps its own descendants. Optimize and rewrite also
run the native CLI beneath the standard-library-only [process supervisor](process_tree.py).
It reaps the complete descendant tree on normal exit, failure or timeout,
including nested loops and workers that create separate process sessions.
Arena's timeout signals the supervisor's process group; the supervisor handles
the descendants that a group signal alone cannot reach. Initialization retains
its own equivalent supervision. Native stdout, stderr and exit status are
preserved. If the supervisor itself receives SIGKILL before reaping finishes,
or the machine fails, cleanup cannot be guaranteed and requires explicit recovery.
This wrapper inherits the existing command's interpreter, arguments, working
directory, environment and streams; it imports no engine code, invokes no shell,
and adds no mounts or downloads. Artifact retention and task checks are unchanged.

### Recovery at the deadline

A campaign killed at the wall never writes its final result, so the search is
read from what the engine published as it ran. The engine publishes each KEEP
before it may start another agent session and defers termination signals across
that publication, so the process group the supervisor signals cannot tear it.
The adapter reads, in order:

| Record | `delivery_selection` |
| --- | --- |
| `forge_experiments/best_result.json` with `correctness_passed` | `timeout_recovered_keep` |
| `forge_experiments/run_state.json` `head_commit` | `timeout_recovered_search_head` |
| The commit Arena accepted before launch | `timeout_recovered_validated_input` |

The KEEP record comes first because it carries the engine's own correctness
verdict. The search head covers a rewrite whose port committed but whose search
never improved on it, and it requires that rewrite to have published `port_ok`;
the engine resolves its own current best the same way. The last row is the
verified input an optimize or initialize campaign already had, which needs no
engine record. A record naming no usable commit is skipped rather than trusted,
and a timeout that reaches none of these still fails with the original candidate
untouched. No published record is deliberately pinned to an engine schema
version, so an engine upgrade cannot silently turn recovery off.

None of these rows is a verdict or a claim of improvement. They identify a
committed bundle, read through `git ls-tree` rather than off the killed
campaign's working tree, for Arena to compile, check and measure normally. The
status keeps `timed_out` alongside the selection, so a reader can tell a
recovered candidate from one the engine chose and reported itself.

### Framework apply-back

The rewrite CLI ends with a stage that patches the operator back into the
framework repository it was ported from, and folds that patch into its own
`success`, so it exits nonzero whenever the patch fails. Arena requests no such
patch: it delivers a standalone candidate bundle and performs its own
acceptance. The stage cannot be switched off, because the engine decides it is
needed from the campaign workspace having a resolvable Git HEAD, which its agent
sessions require before any port attempt can start.

An engine whose capability probe reports `applyback_optional` is asked to skip
it, which also returns the reserve it would otherwise hold to the search. The
option is passed only where the probe reports it, because an engine that does
not define it would reject the campaign before it starts.

Where the stage still runs, a rewrite whose result reports a completed port, no
failure class, and that patch as its only failed stage keeps its outcome, and
the adapter records `applyback_not_requested` with the engine exit code and
error. A nonzero exit with any other explanation still fails the campaign.
Delivery always reads `flydsl_best_commit`: once the patch commits,
`best_commit` names that commit, whose tree carries framework edits rather than
the attempt's bundle.

On such an engine two costs remain. It reserves the last 20 minutes of whatever
budget it receives for the stage, so a rewrite searches for that much less than
`timeout_seconds` suggests, and the stage edits the engine's own copy of any
framework sources the task materialized. That copy is disposable and never
scored, because Arena evaluates the delivered bundle in its own workspace.

One `timeout_seconds` budget covers setup, preflight, initialization and optimization.
Commands receive the same absolute deadline, and every task action is capped by
its remaining time. The rewrite's existing nested loop is not followed by a
second outer loop. A Linux subreaper cleans up descendants even when the engine,
SDK or runner starts a separate process session.

Inside that budget the engine owns its own round admission and reserves. The
adapter hands it one relative budget, already short of the campaign by the
startup margin and floored where the engine floors its own, and caps each task
action by the time left when that action starts. Initialization runs under its
own smaller phase deadline. Task tolerances, cases, measurements and canonical
checks remain mandatory for every accepted candidate, and an intermediate KEEP,
a recovered candidate, or an initialization-only run is not an Arena completion
verdict.

Common Arena post-processing owns exports. Neither this launcher nor the alias
backfills an SIKL solution or assumes `kernel.py` / `workload.json`. The obsolete
agent-specific backfill utility has been removed; exports use the framework.

## Installed KernelForge runtime

The command line and the measurement driver are the whole interface to the
engine. The adapter patches no engine symbol and pins no engine module, so the
two release cadences are independent; [engine.py](engine.py) exists for the two
jobs that genuinely need the engine importable. `--arena-probe` reports what the
installed engine accepts, in the exact interpreter the campaign will use, and
checks that both commands still take the options the adapter builds.
`--arena-initialize` runs the correctness-only implementer the engine publishes
only as a library. It refuses anything else.

Upstream owns its GPU DSL/compiler support and SDKs. Install the `forge` extra in
the GPU runtime environment (it also pulls the agent SDK dependencies):

```bash
# HYPERLOOM_ROOT names the separately mounted/pinned checkout in the container.
python3 -m pip install "${HYPERLOOM_ROOT}[forge]"
python3 agents/forge/engine.py --arena-probe
```

The checked dependency metadata requires Python 3.10+, Click, PyYAML, Anthropic,
Claude Agent SDK and OpenAI Codex SDK. `torch` is deliberately supplied by the
ROCm image; do not install an arbitrary PyPI Torch wheel over it. FlyDSL must be
present for FlyDSL tasks. Provider authentication must be available to the
selected backend; existing authentication environment values are preserved.
A dedicated interpreter can be selected with run-level `agent.python`.

Backend selection follows validated `candidate.language`. Operator and source
owner identity come from `kernel_identity`; a CK operator written in HIP keeps
its CK operator identity while using the HIP backend. The installed engine
provides no TileLang backend. TileLang tasks remain valid Arena tasks, but Forge
rejects them explicitly; it never substitutes FlyDSL. The retained legacy backend
helper also rejects missing registries and unsupported languages.

New scratch repositories start on `codex/arena-forge`, independently of the
runtime's Git default branch; upstream refuses to optimize on `main` or `master`.
The public task assessment and final candidate installation enforce the task
boundary, including symbol-scoped colocated Python harnesses: the bridge checks
the declared symbol boundary against its independent template before
compilation, correctness, or performance, and installation repeats that check.

Search defaults live in [agent_config.yaml](agent_config.yaml); the run may
override `workflow`, `model`, `agent_backend`, `permission_mode`,
`timeout_seconds`, `session_timeout_seconds`, `max_port_attempts`,
`initialization_max_attempts`, `initialization_budget_fraction`,
`supervisor_backend`, and `python`. An empty supervisor follows the selected
backend. Lane count, task preparation, profiling, planning probes and knowledge
warm starts follow the engine defaults. Arena publishes the task and leaves the
search policy to Forge. Two consequences are load bearing. The public task
protocol exposes no profiler invocation, so the bridge answers `--profile-run`
with an explicit unsupported capability. A single-file upstream recipe cannot
describe a multi-file candidate, so a task whose `candidate.editable` spans more
than one file may receive a warm start that covers only part of its submission.

Search behaviour stays behind what the removed patch layer forced until the
engine grows the seams it took: no injected protected paths, no total wall clock
on an implementer invocation, no pluggable correctness adjudication, upstream's
own seed and PORT prompt, and the apply-back stage attempted rather than
declined.

## Verification and framework integration

```bash
python3 -m pytest -q tests/test_forge*.py
# Additionally exercise real installed upstream orchestration without LLM/GPU:
AKA_FORGE_PROBE_PYTHON=<python-with-hyperloom-forge> \
  python3 -m pytest -q tests/test_forge_v2.py tests/test_forge_initialization.py
```

Most tests drive the adapter with a replaced engine subprocess: they cover the
protocol, the delivery selection and the recovery paths, and their CPU timing
fixtures are not GPU measurements or actual optimization runs. The tests gated on
`AKA_FORGE_PROBE_PYTHON` use the installed engine for real — that a driver's
failure diagnostics are never read as a benchmark or a correctness pass, that the
scratch branch survives campaign preflight under a `main` or `master` Git
default, and that HIP/Triton initialization runs the real Forge implementer,
session execution, workspace guard and correctness gate. They cover rejected
attempts, an unedited workspace, harness tampering, session timeouts and a
failing loop. The provider is scripted throughout, so they prove neither model
quality nor GPU compiler compatibility. Real LLM/GPU qualification remains a
required integration gate, including at least five tasks of each retained task
type in the parent campaign. HIP/Triton initialization is implemented, but that
campaign is required before claiming those task types are fully qualified.

The framework integration must:

1. Generate the protected context after initial TaskSession validation and pass
   it to the launcher. Preserve it outside agent-editable workspaces.
2. Register `forge` as the public template. If retaining the old template,
   route it to the same launcher and common post-processing.
3. Catch `ForgeRunError` as a failed agent invocation, retain diagnostics, and
   independently evaluate any delivery through the common evaluator.
4. Perform shared exports only after the final accepted candidate and results
   are available. Never restore the alias's old SIKL backfill call.

Remaining private legacy helpers in `common.py`, `launch_agent.py` and
`drivers/arena_task_adapter.py` are compatibility imports only. The public v2
launcher does not call their legacy task dispatch or report parsers.
