# quality_loop agent

`quality_loop` is a repository-level task curator. It audits every selected task,
attempts one repair for blocking validator failures, runs exactly one Codex
optimization iteration, sends the result to an independent Codex reviewer, and
optionally hardens an easy baseline or task cases behind fail-closed correctness
gates. It records unrepairable task defects locally and bundles all accepted task
changes into at most one draft pull request. It never creates GitHub issues.

Unlike normal Arena agents, `quality_loop` is not registered in `AgentType`.
Normal launchers operate once inside one copied task workspace; this workflow owns
the full repository campaign, isolated git worktree, resume manifest, and final PR.

## Hard preflight

A real run stops before creating a branch or modifying a task unless all of these
pass:

- `gh auth status -h github.com`
- the authenticated account has repository write permission
- `git`, `gh`, and `codex` are installed
- Git has a usable author identity for task commits
- the source worktree is clean
- the configured GPU/runtime is available through the Docker runner

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
  QUALITY_LOOP_ARGS="--tasks hip2hip/gpumode/GELU triton2triton/vllm/triton_rms_norm"
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

Tasks use the common [v2 task contract](../../docs/how-to/add-task.md).
Discovery preserves the full path below `tasks/` as the stable ID, including when
copying a task to a scratch directory. Nested dependency configs are not separate
tasks. `TaskSpec` validates every selected or edited config; legacy task fields
are rejected rather than silently translated.

Language, starting state, candidate paths and baseline policy come from
`candidate` and `baseline`. Directory names select tasks only. Commands and their
timeouts come from `spec.action(...)`. The task runner continues to own its
reference, cases, comparison rules and timing implementation. Configs written by
promotion are serialized through `TaskSpec.to_mapping()` and remain v2.

Optimizer/repair/case-enhancement (`backend`), reviewer and validator have
independently configurable model and timeout budgets. Defaults use the previously
smoke-tested `gpt-5.6-terra` with `medium` effort; see
[CLI qualification](../../docs/reference/agent-model-defaults.md) for the dated
model evidence. The role defaults are in [agent_config.yaml](agent_config.yaml).
For example, a run config can override just the validator budget:

```yaml
quality_loop:
  validator:
    name: codex
    model: gpt-5.6-terra
    effort: medium
    timeout_seconds: 1800
```

An omitted validator block retains its own defaults instead of inheriting a more
expensive optimizer. Reviewer settings inherit an explicitly configured backend
when no reviewer block is supplied. CLI nonzero exits, failed/missing completed
turn events and timeouts are operational failures; timeout cleanup terminates the
CLI process group. These failures do not authorize task edits.

## Per-task gates

1. Run the shared task validator in a fresh workspace. Require the framework's
   completion marker, matching report hash, task ID, fresh timestamp and successful
   framework status. Missing or operationally failed reports stop the attempt.
2. Record WARN findings without repairing them. For task FAIL, allow one repair
   and re-run the validator in a new workspace; a repair needs clean PASS.
3. Use the shared `TaskSession` to validate the initial package and freeze its
   baseline before the optimizer edits anything. Unimplemented candidates remain
   legal only during initial validation. The declared baseline correctness policy
   still applies; a diagnostic baseline never relaxes final candidate correctness.
4. Run one Codex optimization iteration, enforce declared edit scopes through the
   shared harness guard, and submit the candidate to the common evaluator.
5. Run an independent read-only review of the current finalized result. The review
   cannot override deterministic failures or absent required tool evidence.
6. Measure the same candidate against the same frozen baseline for the configured
   number of confirmations. The easy-task gate requires finite speedups, matching
   case counts, consistent methods and successful candidate/tool/reviewer gates.
7. Promote a baseline only when it is an implemented `initial_candidate` with
   committed editable sources. Preserve nested paths and tree helpers. A provided
   baseline is independent: copying a candidate cannot replace its implementation.
   After cross-language promotion, declare the new starting/baseline language.
8. Case enhancements may edit test/harness paths or the declared
   `evaluation.workloads` file, never candidate scopes. Revalidate the original
   baseline with new cases and check the actual optimized candidate. An original
   generation stub is not executed as a candidate. Failed hardening is rolled back.
9. Every material task change needs a fresh framework-finalized validation PASS
   before it is applied to the audit worktree. WARN does not authorize publication.
10. Before any host commit, recheck retained validation evidence and task file
    fingerprints. The complete diff must match accepted per-task paths in
    `state.yaml`; unexpected edits abort publication.

Each attempt has a unique artifact directory. Existing workspaces, reports and
rejected candidates are preserved; a reused output path moves to a sibling
`.quality_loop_history/` directory rather than being deleted. Old reports are not
copied into new task packages. Resume skips a terminal task only while its files
and completed validation evidence still match the recorded fingerprints.

Run artifacts are written under `quality_loop_runs/<run-id>/`; the isolated audit
branch lives under `.quality_loop_worktrees/<run-id>/`. Both are ignored by Git.
Tasks pinned to another GPU architecture are reported as `platform_deferred`; run
the matching campaign to audit them. No accepted changes means no empty PR.

## Shared-runtime integration

[runtime.py](runtime.py) is the only quality-loop lifecycle adapter. It calls
`TaskSession.create(spec, workspace, state_directory)`, `validate_initial()` and
`candidate_action(...)`. Initial baseline evidence and `agent_context.json` stay
in the session state directory outside the candidate workspace. Confirmation
measurements reuse this same session, never snapshot an optimized candidate as
its own baseline.

The shared runtime provides:

- v2 materialization through `src.preprocessing.setup_workspace`, including all
  declared `workspace.sources`, without modifying committed task packages;
- v2 prompting with the stable `_task_id` supplied by quality loop;
- `src.evaluator.evaluate_task_session(session, *, eval_config, logger)`, which
  uses that session's frozen baseline and manifest, performs all candidate and
  configured evaluation-tool gates, scores through the shared scoring code,
  writes a fresh `session.workspace / "task_result.yaml"`, and returns the same
  report mapping (including `task_name`, case counts, method consistency,
  compilation/correctness/tool gates and speedup);
- `src.task_run.validate_task_session`, which executes initial actions, passes
  captured evidence to the validator, and re-finalizes its semantic review with
  the original in-memory context. Validator model/effort and timeout settings
  remain independent of the optimizer.

A missing session/evaluator entrypoint raises an actionable error. There is no
fallback to legacy commands, local scoring or a model-authored PASS. The focused CPU tests use
synthetic timing rows only to exercise protocol plumbing; they are not GPU timing
or task validation evidence.

## Focused verification

The worker change was tested on Linux with Python 3.12.3 and pytest 9.1.1:

```bash
python3 -m pytest -q tests/test_quality_loop.py tests/test_quality_loop_v2.py tests/test_quality_loop_backend.py
python3 -m compileall -q agents/quality_loop
git diff --check
```

Against worker base `deefc493`, 51 tests and three subtests passed; seven shared
lifecycle/prompt tests skipped because those parent-owned modules postdate the
base. Loading the actual shared `src` modules from integration commit `d88c9c55`
produced 58 passed tests, three passed subtests and no skips. Subsequent integration
coverage in `tests/test_task_run_v2.py` exercises the actual shared scoring and
validator paths with CPU processes and synthetic timing rows. The CLI
tests use local fake processes, including a child-process timeout test; no new
paid inference or GPU work was submitted in this change.
