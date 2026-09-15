# Forge agent

Forge is one agent integration for the [public task v2 contract](../../docs/how-to/add-task.md).
The task declares its candidate, baseline, entrypoints and evaluation commands.
It does not supply a Forge driver or consume `KERNELFORGE_*` variables. The old
`forge_operator2flydsl` template delegates to this same launcher and configuration.

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
whose config originally declared a stub does not repeatedly PORT. A broken
changed candidate or failing environment is an error, not an implicit stub.
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

### HIP/Triton initialization

The examined loop can continue after its initial candidate benchmark fails when
external baseline timings exist, but its first incumbent then defaults to those
baseline timings. A first correct implementation slower than that baseline can
be rejected as an optimization. Initialization therefore uses the existing Forge
correctness-only implementer before entering the ordinary performance search:

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
   candidate as its first incumbent, even when its initial speedup is below 1x.

The first valid implementation need not beat the production baseline. Its
acceptance says nothing about final Arena scoring. If a successfully completed
loop reports no best commit because no iteration earned KEEP, delivery can select
the recorded initial correct commit; its status says
`delivery_selection: initial_correct_implementation`. The loop's measurements and
iteration count remain unchanged. An engine error or timeout still fails the run.

The phase has a bounded attempt count and uses at most
`initialization_budget_fraction` of the remaining campaign budget, reserving the
rest for the loop. Each provider session also honors `session_timeout_seconds`.
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
A timeout, failed PORT, nonzero engine exit or missing structured result reports
failure and leaves the original task candidate unchanged. Diagnostic scratch,
logs, result JSON and selected artifact hashes are preserved in the fresh Forge
artifact directory; old experiment directories are never removed.

One `timeout_seconds` budget covers setup, preflight, initialization and optimization.
Commands receive the same absolute deadline, and every task action is capped by
its remaining time. The rewrite's existing nested loop is not followed by a
second outer loop. A Linux subreaper cleans up descendants even when the engine,
SDK or runner starts a separate process session.

Common Arena post-processing owns exports. Neither this launcher nor the alias
backfills an SIKL solution or assumes `kernel.py` / `workload.json`. The legacy
`solution_backfill.py` utility is no longer called by either agent path; its
replacement belongs in the framework's export implementation.

## Installed KernelForge compatibility

`upstream.py` is an explicit, process-local compatibility layer. It changes no
shared Hyperloom source or installed package. It keeps the two real upstream CLI
entrypoints and supplies the following missing adapter capabilities:

- Replace the upstream legacy `compile_command` / `correctness_command` reader
  with the public v2 bridge, including nested-loop canonical acceptance.
- Use the task's actual entrypoints and instructions for PORT, replacing the
  upstream derived-builder prompt and seed convention.
- Carry and commit the complete candidate bundle; admit new nested helpers only
  inside declared candidate boundaries.
- Launch the nested loop through this same compatibility layer.
- Mark upstream-framework apply-back as **not requested**, rather than claiming
  that a framework patch passed. The corresponding 20-minute reserve is removed;
  Arena receives standalone task artifacts and performs final acceptance.

The preflight verifies the package version, exact reviewed module hashes in
[upstream_compatibility.json](upstream_compatibility.json), CLI options and Python
hook signatures **before installing any compatibility hook**. It refuses a
different version or changed implementation even when the signatures still fit.
A source checkout without distribution metadata must match the same hashes.
There is no bypass switch. A new release requires a review of the affected private
interfaces, updated compatibility tests and a GPU qualification campaign before
updating the pin; do not simply regenerate hashes to suppress a mismatch.
The accepted version, reviewed commit and hashes are recorded in
`arena_forge_status.json`. No shared Hyperloom files are modified.

The adapter also binds the implementer's working directory to the engine root,
protects tracked files outside the candidate bundle, and lets the SDK workspace
guard recognize new files inside declared candidate scopes. These are scoped
process-local changes to the pinned Forge factory and run-spec constructor.
The public task assessment and final candidate installation still enforce the
task boundary, including symbol-scoped colocated Python harnesses.

Compatibility was inspected and CPU-tested against Hyperloom commit
`0425bde3f6e76e1588400c37d056dfd3bb75ac11`, package version `1.1.0`.
Upstream owns its GPU DSL/compiler support and SDKs. Install the `forge` extra in
the GPU runtime environment (it also pulls the agent SDK dependencies):

```bash
# HYPERLOOM_ROOT names the separately mounted/pinned checkout in the container.
python3 -m pip install "${HYPERLOOM_ROOT}[forge]"
python3 agents/forge/upstream.py --arena-probe
```

The checked dependency metadata requires Python 3.10+, Click, PyYAML, Anthropic,
Claude Agent SDK and OpenAI Codex SDK. `torch` is deliberately supplied by the
ROCm image; do not install an arbitrary PyPI Torch wheel over it. FlyDSL must be
present for FlyDSL tasks. Provider authentication must be available to the
selected backend; existing authentication environment values are preserved.
A dedicated interpreter can be selected with run-level `agent.python`.

Search defaults live in [agent_config.yaml](agent_config.yaml); the run may
override `workflow`, `model`, `agent_backend`, `permission_mode`,
`timeout_seconds`, `session_timeout_seconds`, `max_port_attempts`,
`initialization_max_attempts`, `initialization_budget_fraction`,
`supervisor_backend`, and `python`. An empty supervisor follows the selected
backend. This adapter currently uses one lane and disables profiling/probes
because the public task protocol provides no profiler invocation. Knowledge
warm starts and publication are disabled for this integration so single-file
upstream recipe formats cannot silently truncate a multi-file submission.

## Verification and framework integration

```bash
python3 -m pytest -q tests/test_forge*.py
# Additionally exercise real installed upstream orchestration without LLM/GPU:
AKA_FORGE_PROBE_PYTHON=<python-with-hyperloom-forge> \
  python3 -m pytest -q tests/test_forge_v2.py tests/test_forge_initialization.py
```

The compatibility test uses real upstream seed/preflight/rewrite dispatch and
canonical acceptance, with controlled PORT/OPTIMIZE replacements. CPU timing
fixtures are protocol tests; they are not GPU measurements or actual optimization
runs. The HIP/Triton tests execute the real Forge implementer, session execution,
workspace guard and correctness gate with a scripted provider, then parse the
actual loop CLI and exercise its incumbent measurement. They cover rejected
attempts, a slower correct initialization, nested new helpers, false success
text, harness tampering, timeouts and unreviewed engine rejection. The loop
callback is controlled in those tests; they do not prove model quality or GPU
compiler compatibility. Real LLM/GPU qualification remains a required integration
gate, including at least five tasks of each retained task type in the parent
campaign. In particular, HIP/Triton initialization is now implemented, but that
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
