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
| Initialization to another target language | Explicit unsupported-capability error |

`workflow: auto` is the default. `optimize` requires a currently verified target
implementation; `rewrite` requires initialization. A changed candidate is rebuilt
and checked against the final target contract before routing, so a resumed task
whose config originally declared a stub does not repeatedly PORT. A broken
changed candidate or failing environment is an error, not an implicit stub.
The backend comes from `candidate.language` and is checked against the actual
installed KernelForge registry. There is no task-name dispatch, suffix inference,
or silent substitution of an unsupported backend with FlyDSL.

The current rewrite engine supports FlyDSL initialization. This integration does
not claim that it can generate HIP or Triton implementations from empty targets.
Existing HIP/Triton/FlyDSL implementations can enter the loop when that installed
engine registers their backend and the task's final checks pass.

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

Both workflows run in a fresh sibling directory of the task workspace. The
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

One `timeout_seconds` budget covers setup, preflight, PORT and nested OPTIMIZE.
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

The preflight verifies required CLI options and Python hook signatures before
starting a campaign. It records the installed package version and source hashes
in `arena_forge_status.json`. These are private upstream interfaces: changes to
KernelForge require rerunning the compatibility tests and GPU campaign, not just
assuming a newer package remains compatible.

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
`supervisor_backend`, and `python`. An empty supervisor follows the selected
backend. This adapter currently uses one lane and disables profiling/probes
because the public task protocol provides no profiler invocation. Knowledge
warm starts and publication are disabled for this integration so single-file
upstream recipe formats cannot silently truncate a multi-file submission.

## Verification and framework integration

```bash
python3 -m pytest -q tests/test_forge_v2.py tests/test_forge_operator2flydsl.py
# Additionally exercise real installed upstream orchestration without LLM/GPU:
AKA_FORGE_PROBE_PYTHON=<python-with-hyperloom-forge> \
  python3 -m pytest -q tests/test_forge_v2.py -k installed_upstream
```

The compatibility test uses real upstream seed/preflight/rewrite dispatch and
canonical acceptance, with controlled PORT/OPTIMIZE replacements. CPU timing
fixtures are protocol tests; they are not GPU measurements or actual optimization
runs. Real LLM/GPU qualification remains a required integration gate.

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
