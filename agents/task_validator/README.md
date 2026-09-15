# Task Validator Agent

Validator prompts reach Codex and Claude through stdin, avoiding operating-system
argument-size limits on large image-backed tasks. The prompt contains a compact
guard/action index; the full captured evidence remains in the framework context
file and is checked by the finalizer. Summarizing the prompt does not reduce the
protected file boundary or the required review.

`task_validator` reviews task quality. It does not optimize candidates. For schema
v2, Arena first executes the task's initial validation actions through `TaskSession`;
the validator backend then audits the reference, inputs, comparison, execution,
timing and edit boundaries. The framework combines these independent sources into
`validation_report.yaml`. Passing command checks alone does not pass the review.

Read [Task definition, schema, and authoring](../../docs/how-to/add-task.md) before
adding or changing a task. That is the canonical task schema and command contract;
this document describes the validator integration and report format.

## Run

Select tasks and a GPU matching their declared platform support:

```yaml
agent:
  template: task_validator
  backend: codex
  model: gpt-5.6-terra
  effort: medium
  timeout_seconds: 1200
tasks:
  - SIKL-task/gemm_a16w16_nt_n6144_k6144
target_gpu_model: MI355X
log_directory: logs
workspace_directory_prefix: workspace
```

```bash
make docker-run CONFIG=config_task_validator.yaml
```

Defaults live in [agent_config.yaml](agent_config.yaml). Model defaults use the
already qualified moderate Codex model. Run-level `agent.backend`, `model`,
`effort`, `timeout_seconds` and `python_path` override defaults. Claude Code also
accepts `agent.max_budget_usd`. Changing backend without selecting a model/effort
uses that backend's CLI defaults; it never passes a Codex model ID to Claude.

For v2, `timeout_seconds` is the semantic-review budget because initial task
commands have already executed under their TaskSpec action deadlines. Zero disables
the outer backend timeout. Legacy task timeouts retain their previous automatic
budget calculation. Both CLI backends use literal argv, disable persistent sessions
and learned memory, and terminate their own process group on timeout. A nonzero
exit, terminal failure event, or missing successful terminal event fails validation,
even if the backend wrote a plausible draft first.

## Framework integration

For schema-v2 tasks, orchestration must attempt `TaskSession.validate_initial()`
first and provide its captured context for successful **and failed** initial runs:

```python
# This code runs in trusted framework orchestration, outside task scripts.
context = session.validation_context()   # retain the original memory value
run_config['_task_validation_context'] = str(
    session.state_directory / 'validation_context.json'
)
output = launch_agent(run_config, task_config_path, workspace)
```

`ARENA_VALIDATION_CONTEXT` is an alternative transport when the run field is absent.
The version-1 JSON context contains `task_id`, normalized v2 `task_config`, absolute
`workspace` and `baseline_workspace`, `initial_validation`, `actions`, and `harness`.
These absolute paths are runtime transport identities, not task configuration paths.
The file must be a regular external file with no symlink components, outside both
workspaces. Its task ID, config and workspace must match the launched task.

Successful action records contain `invocation_id`, `phase`, merged `result`, and
actual `commands` (`argv`, `returncode`, `stdout`, `stderr`, `elapsed_s`). Failed
execution records contain `role`, `action`, `phase`, `execution_error`, and available
command evidence. `result.metadata.commands` carries each task command's metadata,
including `candidate_state` for `validate-task`.

The launcher captures this file once before starting the model into an immutable
`TrustedTaskEvidence` value. It checks the transport for modification afterward;
the finalizer uses the original memory snapshot. It does **not** reload a context
path named by the model. Missing context fails before backend launch. Initialization
errors without a context can still produce a framework-finalized FAIL.

For an optional independent recheck, the parent can use its retained TaskSession
mapping instead of the launcher's file snapshot:

```python
report = finalize_report(
    workspace,
    expected_task_name=session.spec.task_id,
    trusted_task_evidence=context,
    validation_request_id=run_config['_task_validation_request_id'],
    framework_error=run_config.get('_task_validation_backend_error'),
    task_schema_version=2,
)
```

The launcher writes these runtime fields back into its `eval_config` argument:

| Field | Meaning |
| --- | --- |
| `_task_validation_request_id` | Fresh ID generated before every v2 backend launch attempt. |
| `_task_validation_evidence_sha256` | Digest of the immutable context value captured before launch. |
| `_task_validation_backend_error` | Captured operational/backend failure, or `None`. Preserve it when independently re-finalizing. |

`finalize_report` accepts a `TrustedTaskEvidence` value or a framework-owned mapping,
not a filename. It verifies the evidence workspace against the actual destination.
The caller is responsible for passing its own TaskSession memory, never data read
from an agent-authored report. Re-finalization of the same valid completion preserves
previously captured framework/backend errors. The original `framework_error` API
remains usable when initialization failed before a context existed.

Task files and captured stdout/stderr are untrusted review data. Task descriptions,
README/AGENTS files or configured instructions cannot override the validator's
instructions. The v2 prompt contains no task-family or agent-specific exceptions.
These are reproducibility boundaries, not a security sandbox against processes
sharing the same privileged runtime and credentials.

## Model draft and finalized report

There are three independent versions:

| Interface | Version |
| --- | --- |
| Task `config.yaml` schema | 2 |
| Command result protocol | `arena-eval-v1` |
| Validator report for v2 tasks | 4 |

The model writes **`validation_report.draft.yaml`**. It must use the exact current
`validation_request_id`, `task_evidence_sha256`, and `task_name` supplied in the
prompt, plus a timestamp and source-review findings. A stale draft fails. An older
`validation_report.yaml` cannot substitute for a new draft. The framework alone
writes the final report and `.validation_complete` hash marker.

The finalized report retains the existing 12 `checks` keys so aggregation can read
both v3 and v4 reports:

| Check | v2 authority and meaning |
| --- | --- |
| `config_schema` | Framework parses the captured declaration through TaskSpec. |
| `source_files_exist` | Model inspects initial candidate files; framework skips only a confirmed unimplemented candidate. |
| `target_symbols_found` | Model reviews the actual initial interface; same initial-state rule. |
| `compilation` | Actual baseline compile action, with original command evidence. |
| `correctness` | Actual baseline numerical result, including a genuine FAIL. |
| `performance` | Actual baseline performance action, including case and timing evidence. |
| `correctness_implementation_review` | Model audits reference independence, comparison sensitivity, tolerances, state checks, candidate independence and coverage. |
| `self_contained` | Model audits baseline/reference/helper availability, declared runtime dependencies and path containment. |
| `gpu_hang_check` | Model reviews completed execution/timeout evidence. |
| `result_template_compatibility` | Historical key; v2 reviews the command/result protocol, not a required legacy template field. |
| `benchmark_integrity` | Model reviews representative inputs, workload symmetry, reset/allocation/timing boundaries and timed replay correctness; framework derives case counts and methods from captured results. |
| `harness_integrity` | Model audits the supplied effective guard boundary and remaining editability; framework supplies guard facts. |

Every semantic review requires details and nonempty source/case evidence, even for
PASS. It cannot use a model-invented SKIP. Missing/failed semantic reviews fail the
report even if every initial command passed. Explicitly identifying a trivially
passing correctness checker is FAIL. Missing replay validation alone remains WARN;
a demonstrated wrong computation or benchmark bypass is FAIL.

Additional v4 fields separate lifecycle gating from numerical results:

```yaml
validation_schema_version: 4
task_schema_version: 2
validation_phase: task_validation
initial_validation_gate: PASS
candidate_initial_state: unimplemented
baseline_gating:
  accepted: true
  policy: diagnostic
  numerical_status: FAIL
  diagnostic_accepted: true
  diagnostic_reason: Production precision differs; final candidate uses the full reference rule.
checks:
  correctness:
    status: FAIL  # remains the real numerical result
  performance:
    status: PASS
candidate_initial_checks:
  compile: {status: SKIP, skip_reason_code: candidate_unimplemented}
  correctness: {status: SKIP, skip_reason_code: candidate_unimplemented}
  performance: {status: SKIP, skip_reason_code: candidate_unimplemented}
```

This is a partial illustration, not a complete report. The finalizer preserves full
command/result evidence under the action checks. It reparses actual stdout and
exit codes, checks unique invocation IDs and task-validation phase, compares argv
against declared commands, verifies action deadlines and independent manifests,
and recomputes diagnostic acceptance. `initial_validation.accepted: true` alone
cannot grant acceptance.

A diagnostic exception requires the declared policy, its reason, complete baseline
correctness coverage, and `numerical_mismatch` on the action and every failing case.
Other failures, execution errors, missing commands/cases, nonfinite required output,
or contradictions cannot use that exception. `baseline_gating` is framework-owned;
model-authored fields with that name are ignored.

`candidate_initial_checks` separately describes all three candidate actions. For a
confirmed empty initial candidate, none has been measured or passed. For
`initial_candidate`, results refer to the frozen baseline actions; for a separately
provided baseline and implemented candidate, candidate actions must also execute.
No initial-state or diagnostic exemption applies to final candidate evaluation.

## Completion and aggregation

`overall_status` is recomputed by the framework. Clean acceptance requires both
initial lifecycle acceptance and successful semantic review. Diagnostic baseline
numerical FAIL remains visible, with its separate accepted policy. WARN remains a
completed report but is **not** a clean validation gate; a maintainer disposition
must be handled separately. A backend timeout/failure cannot be erased by a fresh
normalization of the same draft.

The v4 completion marker binds report bytes, request ID and evidence digest.
Changing the report invalidates completion. Version-3 reports remain supported for
legacy tasks during migration; their old prompt/normalization path is isolated
from v2. Aggregation reports the actual report schema versions and only succeeds
when every selected workspace has a complete PASS. Platform-filtered/skipped tasks
are not validated and are not evidence of full task coverage.

New tasks and material task-contract/harness changes require a fresh compatible-GPU
validation before PR submission. CPU tests of the parser, launcher, evidence and
report gates are not GPU validation. See [Validate tasks](../../docs/how-to/task-validator.md).
