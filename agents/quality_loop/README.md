# quality_loop agent

`quality_loop` is a repository-level task curator. It audits every selected task,
audits it with the [QA auditor](../../docs/how-to/qa-auditor.md), attempts bounded
repairs for blocking findings, runs exactly one Codex optimization iteration, sends
the result to an independent Codex reviewer, and optionally hardens an easy baseline
or task cases behind fail-closed correctness gates. It records unrepairable task defects locally and bundles all accepted task
changes into at most one draft pull request. It never creates GitHub issues.

Unlike normal Arena agents, `quality_loop` is not registered in `AgentType`.
Normal launchers operate once inside one copied task workspace; this workflow owns
the full repository campaign, isolated git worktree, resume manifest, and final PR.

## Hard preflight

Host initialization stops before creating the audit branch unless these checks
pass:

- `gh auth status -h github.com`
- the authenticated account has repository write permission
- `git`, `gh`, and `codex` are installed
- Git has a usable author identity for task commits
- the source worktree is clean

The Docker execution phase checks the selected GPU/runtime before task work.
Successful host initialization alone is not evidence of GPU execution.

Only the host-side deterministic publisher uses `gh`. The Docker runner performs
GitHub preflight and creates the audit worktree on the host, runs Codex/GPU work
without mounting GitHub credentials, then returns to the host to commit accepted
task changes, push, and open the draft PR. The main checkout is
read-only inside that container; only this run's artifact and isolated worktree
directories are writable. Codex login state is copied into an ephemeral container
home instead of being writable in place.

## Run

Inspect task selection without credentials, GPU work, or mutations:

```bash
python3 -m agents.quality_loop \
  --config example_configs/quality_loop_mi300.yaml \
  --plan
```

Run through the supported Docker environment:

```bash
make docker-quality-loop CONFIG=example_configs/quality_loop_mi300.yaml
```

Use `example_configs/quality_loop_mi355x.yaml` on MI355X (`gfx950`).

Limit a smoke run to several tasks:

```bash
make docker-quality-loop \
  CONFIG=example_configs/quality_loop_mi300.yaml \
  QUALITY_LOOP_ARGS="--tasks triton/normalization/rms_norm triton/gemm"
```

Resume after interruption:

```bash
make docker-quality-loop \
  CONFIG=example_configs/quality_loop_mi300.yaml \
  QUALITY_LOOP_ARGS="--resume 20260803_120000"
```

`--no-publish` still requires the GitHub login/write preflight, but suppresses
push and PR creation. `--plan` is the only intentionally offline mode.

## Task contract and budgets

`quality_loop` works only on schema-v3 kernel task packages, the
[canonical task contract](../../docs/how-to/add-task.md): `task.yaml`,
`workloads.json`, `reference.py`, the vendored `baseline/`, an optional `inputs.py`
and `README.md`. Discovery finds `tasks/**/task.yaml`, keeps the full path below
`tasks/` as the stable ID (also for scratch copies), skips nested `task.yaml` files
and ignores schema-v2 packages such as the e2e serving tasks. Every selected or
edited package is loaded through `src.task_contract`; an invalid package is
rejected rather than adapted.

Each audit materializes the package for its audit language
(`src.task_contract.audit_target_language`: the baseline's language when it is a
target language), so the candidate starts as a copy of the baseline and the audit
also checks the candidate path. The optimization language is the top-level
`target_language` of the quality_loop config (`triton`, `gluon`, `hip` or
`flydsl`); `null` optimizes each task in its audit language. Workspaces get the
framework-generated `config.yaml`, `_aka_eval/` and `solution/`; only `solution/`
is a candidate, and none of the three is ever proposed as a task change.

Use `quality_loop.mode: repair_only` to audit and repair task quality without
optimizing candidates, promoting baselines or claiming a speedup. That mode has
zero optimization iterations. Normal `optimize` mode retains exactly one.
`max_repair_attempts` defaults to 1 and permits 1--3 attempts. Every attempt needs
a changed, previously unseen proposal; unchanged audit reruns cannot erase an
existing finding. All proposals and rejected evidence remain in the artifacts.

Every blocking QA finding is repaired. Warnings remain recorded by default;
`repair_warnings` can explicitly select QA warning check ids to repair:

```yaml
quality_loop:
  mode: repair_only
  max_repair_attempts: 2
  repair_warnings: [QA-README, QA-BOUND-OVERRIDE]
```

[repair_policy.py](repair_policy.py) keeps scoring unchanged. A repair may add
correctness-only cases, edit `inputs.py`, `reference.py`, `README.md` and the
`task.yaml` description, add a `baseline/LICENSE` and tighten an error bound. It
may not change the `task.yaml` interface, reference or baseline declarations
(language, entry, provenance), operator, platform or timeout, the vendored
baseline sources, any original workload case, or add a performance case; a
`compare` change must provably tighten the bound (same bound with stricter
parameters, or `exact`). An independent repair reviewer compares the source with
the original package, checks reference, input-generation and case semantics not
captured by those deterministic checks, and must close every retained finding. A
new QA PASS alone cannot authorize weakening a task. The fresh audit and the
read-only review both precede acceptance or publication of repaired files.

Optimizer/repair/case-enhancement (`backend`), reviewer and auditor have
independently configurable model and timeout budgets. Defaults use the previously
smoke-tested `gpt-5.6-terra` with `medium` effort; see
[CLI qualification](../../docs/reference/agent-model-defaults.md) for the dated
model evidence. The role defaults are in [agent_config.yaml](agent_config.yaml).
The auditor block configures the QA auditor's semantic review (`codex`,
`claude_code`, or `none` for its deterministic checks only):

```yaml
quality_loop:
  auditor:
    name: codex
    model: gpt-5.6-terra
    effort: medium
    timeout_seconds: 1800
```

An omitted auditor block retains its own defaults instead of inheriting a more
expensive optimizer. Reviewer settings inherit an explicitly configured backend
when no reviewer block is supplied. CLI nonzero exits, failed/missing completed
turn events and timeouts are operational failures; timeout cleanup terminates the
CLI process group. These failures do not authorize task edits.

Optimizer, repair, reviewer and case-enhancement prompts use anonymous stdin,
not command-line arguments. Each invocation logs a unique private receipt directory
beside its task workspace: `.quality_loop-<role>-<unique-id>/`. It contains
`stdout.log`, `stderr.log` and `process.json`, including failed and timed-out calls.
The metadata records the role, model, safe argv, prompt byte count/hash, timestamps,
exit status, observed and retained-prefix hashes and the latest terminal/usage events; it does
not serialize prompt text, credentials or the process environment. The QA auditor's
review runs through `agents.backend_cli` and does not use this receipt mechanism.

Raw stdout/stderr retain at most 8 MiB/1 MiB per invocation. Truncation is explicit;
the backend continues draining, hashing and checking terminal events after those
limits. A JSON event line larger than 1 MiB is an operational failure. Available
token counts are CLI-reported evidence, not a dollar-cost estimate. A receipt left
in `starting`/`running`, or without stream EOF, is incomplete evidence. These logs
are outside task edit scopes and must be preserved with run artifacts; permissive
agent subprocesses are not a security sandbox. Role completion and its receipt do
not approve a candidate or replace independent review and framework finalization.
Raw receipts are flushed/fsynced before success; status files are flushed/fsynced
before atomic replacement, followed by a directory fsync. Storage errors fail the
call, with a failed receipt saved when storage permits. A shorter post-exit pipe
drain timeout is identified separately from the configured role deadline.

## Per-task gates

1. Audit the package in a fresh workspace through `src.task_run.audit_task_session`
   with the QA auditor. Accept only the framework-finalized `qa_report.yaml` with
   its `.qa_complete` marker, this request's ID, the task ID, a fresh timestamp, the
   current rubric and the exact digest of the audited package, and only if the audit
   changed no task file. A `QA-AUDIT` finding (the audit itself failed) or a missing
   report is an operational failure that stops the attempt. `status: FAIL` is the
   task judgment.
2. Record warnings; repair only explicitly selected warning checks. For blocking
   findings, attempt the configured bounded repairs with the repair policy and
   finding checks, then re-audit in a new workspace. A repair needs a clean PASS and
   an accepting independent repair review. In `repair_only` mode stop here and
   publish only accepted task repairs through the same host finalizer.
3. Materialize the package for the optimization language and use the shared
   `TaskSession` to validate it and freeze its baseline before the optimizer edits
   anything. An unimplemented template (target language other than the baseline's)
   is legal only during initial validation.
4. Run one Codex optimization iteration in `solution/`, enforce the edit boundary
   through the shared harness guard, and submit the candidate to the common
   evaluator.
5. Run an independent read-only review of the current finalized result. The review
   cannot override deterministic failures or absent required tool evidence.
6. When the first speedup reaches the configured easy-task threshold and the
   reviewer accepts, perform the remaining confirmations against the same frozen
   baseline. The easy-task gate requires finite speedups, matching case counts,
   consistent methods and successful candidate/tool/reviewer gates.
7. Promote a baseline only for a same-language candidate (it started as a copy of
   the baseline). Its `solution/` tree replaces the vendored sources in `baseline/`
   (license files and notes are kept) and `baseline.provenance` records the
   quality_loop run, the candidate digest and the replaced upstream source; the
   upstream license still applies to the derived code. A cross-language candidate
   is never promoted. The original baseline and the optimized candidate must pass
   the dual correctness gate, and the promoted package needs a fresh QA PASS.
8. When enabled and requested by an accepting reviewer on a task whose initial audit
   passed, propose case enhancements. Only `workloads.json` and `inputs.py` may
   change, existing cases stay identical and new cases are correctness-only; any
   other change rejects the whole proposal. The original baseline (prepared through
   the shared session on the new cases) and the actual optimized candidate must
   pass; failed hardening is rolled back.
9. Every material task change needs a fresh framework-finalized QA PASS before it
   is applied to the audit worktree. Warnings do not block; a FAIL does.
10. Before any host commit, recheck retained QA evidence and task file
    fingerprints. The complete diff must match accepted per-task paths in
    `state.yaml`; unexpected edits abort publication.

Each attempt has a unique artifact directory. Existing workspaces, reports and
rejected candidates are preserved; a reused output path moves to a sibling
`.quality_loop_history/` directory rather than being deleted. Old reports are not
copied into new task packages. Resume skips a terminal task only while its files
and completed QA evidence (report digest, package digest and rubric version)
still match the recorded values.

Task proposals exclude the framework-written `config.yaml`, `_aka_eval/` and
`solution/`, QA and evaluation reports, compiled objects/libraries and ELF
executables (including extensionless binaries). Ordinary tensor/input fixtures
remain eligible for review. Reviewer snapshots still observe candidate, evaluator
and report mutations. The host checks for generated outputs again before
committing, even when an older campaign manifest lists them as accepted changes;
it refuses publication without deleting the experiment artifacts. Keep raw results
in the run artifact directory and share summaries and artifact links in the PR
instead of committing outputs.

Run artifacts are written under `quality_loop_runs/<run-id>/`; the isolated audit
branch lives under `.quality_loop_worktrees/<run-id>/`. Both are ignored by Git.
Tasks pinned to another GPU architecture are reported as `platform_deferred`; run
the matching campaign to audit them. No accepted changes means no empty PR.

A completed campaign or accepted implementation review does not mean a baseline
was promoted or cases were enhanced. Inspect `baseline_hardened`, `cases_enhanced`,
accepted changes and retained stage logs separately.

## Shared-runtime integration

[runtime.py](runtime.py) is the only quality-loop lifecycle adapter. It calls
`TaskSession.create(spec, workspace, state_directory)`, `validate_initial()` and
`candidate_action(...)`. Initial baseline evidence and `agent_context.json` stay
in the session state directory outside the candidate workspace. Confirmation
measurements reuse this same session, never snapshot an optimized candidate as
its own baseline.

Before independent review, the controller writes a unique `review-evidence-*.json`
index in that external session directory and supplies its path and SHA256 in the
reviewer prompt. It locates the initial contexts and this evaluation's candidate
compile/correctness/performance records, including actual argv, exit codes, stdout
protocol envelopes and per-case results. `task_result.yaml` is an aggregate; it
does not need to embed those action records. An action without a completed record
is explicitly marked `NO_COMPLETED_ACTION`, never inferred to have passed.

Contexts are captured before optimization. Candidate records must match the
controller's in-memory executions, and the result must bind the current candidate
sources. Candidate-provided evidence paths are not consulted. Indexed files are
checked again after review, as are the frozen baseline and existing workspace
boundaries. The index improves evidence discovery; it does not relax numerical,
coverage, tool, timing or reviewer gates. It is a reproducibility check, not an OS
security sandbox. Preserve it with the external session and role receipts.

The shared runtime provides:

- schema-v3 materialization through `src.preprocessing.setup_workspace(...,
  target_language=...)`, without modifying committed task packages;
- prompting through `src.prompt_builder.prompt_builder` with the stable `_task_id`
  supplied by quality_loop (it reads the workspace's resolved `config.yaml`);
- `src.evaluator.evaluate_task_session(session, *, eval_config, logger)`, which
  uses that session's frozen baseline and manifest, performs all candidate and
  configured evaluation-tool gates, scores through the shared scoring code,
  writes a fresh `session.workspace / "task_result.yaml"`, and returns the same
  report mapping (including `task_name`, case counts, method consistency,
  compilation/correctness/tool gates and speedup);
- `src.task_run.audit_task_session`, which runs the initial actions, passes the
  captured evidence to the QA auditor and verifies afterwards that neither the audit
  nor its review changed protected task state or evidence. Auditor model/effort and
  timeout settings remain independent of the optimizer.

A missing session/evaluator entrypoint raises an actionable error. There is no
fallback to local scoring or a model-authored PASS.

## Focused verification

CPU tests (no GPU, model or network):

```bash
python3 -m pytest -q tests/test_quality_loop.py tests/test_quality_loop_v2.py \
  tests/test_quality_loop_repair_policy.py tests/test_quality_loop_artifacts.py \
  tests/test_quality_loop_review_evidence.py tests/test_quality_loop_backend.py \
  tests/test_quality_loop_backend_evidence.py
```

The audit tests run the real materialization, `TaskSession` and
`audit_task_session` with a fake auditor that finalizes reports through
`agents.qa_auditor.report`; on CPU the initial GPU actions fail and are recorded
as such. The review-evidence tests drive a real `TaskSession` with a small CPU
protocol runner, because a schema-v3 evaluator needs a GPU; their timings are
synthetic. None of these tests is GPU validation or a QA audit of a real task.
