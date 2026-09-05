# AGENTS.md

## Purpose

AgentKernelArena is a controlled environment for evaluating GPU-kernel agents.
Every change must preserve reproducibility, benchmark integrity, meaningful
correctness checks, and fair baseline-versus-candidate comparison.

Read [README.md](README.md) and [CONTRIBUTING.md](CONTRIBUTING.md) before making
broad changes. Use repository-relative paths in code, configuration, docs, and
instructions; never embed paths from a local checkout.

## Repository map and sources of truth

- `main.py`: run orchestration, resume, and parallel scheduling.
- `src/`: shared prompting, workspace setup, evaluation, scoring, and reporting.
- `agents/`: isolated agent integrations and their own configuration.
- `tasks/`: self-contained kernel tasks and task-local runners.
- `example_configs/`: run-level agent, task, and GPU selections.
- `src/tools/perf/`: canonical benchmark helpers materialized into workspaces.
- `src/eval_tools/`: evaluator-side optional analysis-tool control plane.
- `tests/`: framework, integration, and regression tests.

When behavior and prose disagree, inspect the implementation. In particular:

- Agent identifiers and loading: `src/module_registration.py`
- Task contract: the task's `config.yaml`
- Evaluation and scoring: `src/evaluator.py`, `src/performance.py`, `src/score.py`
- Harness protection: `src/harness_guard.py`
- Benchmark helpers: `src/tools/perf/`
- Evaluation-tool policy: `docs/how-to/use-evaluation-tools.md`

Update relevant docs in the same change when behavior or public configuration
changes. Avoid copying volatile model IDs, image tags, or complete registries
into new guidance; link to their source of truth instead.

## Architecture boundaries

Keep agents and tasks independent.

- A task defines a backend-neutral contract through its files and `config.yaml`.
  Task code must not import from `agents/` or depend on a particular agent,
  model, prompt format, provider, or authentication mechanism.
- Agent implementations must not import task modules, mutate committed task
  sources, or special-case task names and paths. Add general capability through
  task configuration and shared framework interfaces instead.
- Keep provider- and CLI-specific behavior under `agents/<agent_name>/`. Move
  code into `src/` only when it is genuinely shared by multiple integrations.
- Isolated tasks must remain self-contained after being copied into a workspace;
  they must not rely on imports from this repository's `src/` tree.

## Working practices

1. Inspect the relevant implementation, config, tests, and docs before editing.
2. Keep the change focused; preserve unrelated user changes in a dirty tree.
3. Add or update a focused regression test for behavioral changes.
4. Run the smallest relevant checks first, then broader checks when practical.
5. Report the commands run, results, hardware used, and checks not run.

GPU experiments are Docker-first. Do not reintroduce a host `python main.py`
workflow. Select a config matching the physical GPU architecture and use:

```bash
make docker-smoke
make docker-check-agents CONFIG=<run-config>
make docker-run CONFIG=<run-config>
make docker-parallel-run CONFIG=<run-config> GPU_IDS=0,1
```

Do not claim GPU validation from code inspection or CPU-only tests. If compatible
hardware, agent credentials, or external dependencies are unavailable, state
that explicitly.

## Task authoring and validation

Follow [the task authoring guide](docs/how-to/add-task.md) and inspect nearby
tasks of the same type.

- Paths in an isolated task must resolve within the task directory. Do not use
  absolute paths, undeclared downloads, or external repositories.
- Command fields in `config.yaml` are lists. Commands must exit nonzero on
  failure and honor their declared timeouts.
- Compilation checks must actually compile or syntax-check the target.
- Correctness checks must compare against a meaningful reference with sensible
  tolerances and representative cases; never accept a text-pattern check or a
  trivially passing harness.
- Performance checks must emit scoreable device timing and preserve equivalent
  work, state, and allocation boundaries for baseline and candidate.
- Repository tasks must declare upstream sources and setup steps explicitly.
- Use `platform_support` for real architecture constraints instead of silently
  skipping cases inside a runner.

Every new task, and every material change to a task contract or harness, must
pass `task_validator` on compatible GPU hardware **before a PR is submitted**:

```bash
make docker-run CONFIG=<validator-config>
```

Require a framework-finalized `validation_report.yaml` whose `overall_status` is
`PASS`. A `WARN` needs an explicit maintainer-approved justification and is not
a clean pass; a `FAIL`, timeout, partial report, stale report, or skipped task
does not satisfy this gate. See
[the validator guide](docs/how-to/task-validator.md) for the current checks and
report contract.

## Benchmark and harness integrity

Never improve a score by weakening expected outputs, tolerances, test shapes,
warmups, sample counts, synchronization, state reset, timing methods, or result
parsing. The centralized evaluator owns `task_result.yaml`; optimization agents
must not fabricate or edit it.

During optimization runs, treat task config, test, script, harness, and configured
performance entrypoints as protected. Some tasks intentionally colocate editable
kernel functions with a protected harness; only the target functions and allowed
implementation helpers are editable. Follow the boundary enforced by
`src/harness_guard.py` rather than assuming the whole file is editable.

Repository-maintenance changes to a harness are allowed only when the task itself
requires them. Such changes need focused regression coverage and a fresh
`task_validator` pass.

## Generated performance helpers

`src/tools/perf/` is the single source of truth for shared benchmark code. Do not
hand-edit committed ROCmBench helper stubs, generated `_aka_benchmark.py` or
`hip_graph_benchmark.hpp` copies, or `AKA-GENERATED` regions in task runners.

After changing canonical helpers, run:

```bash
make check-perf-helpers
```

Run `make sync-perf-helpers` only when stub or marker structure changes. For
inspection, use `make materialize-perf-task TASK=tasks/<task-path>` or
`make materialize-perf-workspace WORKSPACE=<workspace>`.

## Agent integrations and evaluation tools

When adding an agent, update all applicable registration, dynamic loading,
prompt/post-processing routing, configuration, docs, and tests. A
`@register_agent` decorator alone does not make an agent selectable; use
`src/module_registration.py` as the registry source of truth.

Evaluation tools belong to the trusted evaluator control plane, not to agent
skills or prompts. Keep them opt-in, isolated, capability-checked, attested, and
fail-closed as described in `docs/how-to/use-evaluation-tools.md`. Do not claim
that a candidate was analyzed merely because a tool exists or its sidecar ran.

## Verification guide

Choose checks according to the affected area:

```bash
python3 -m pytest -q tests/<relevant-test>.py
make check-docker-runner
make check-evaluator
make check-held-out
make check-visualization
make check-perf-helpers
```

Use `python3 -m pytest -q tests` for the broad non-GPU suite when the environment
has the required dependencies. Changes under `src/eval_tools/` should include
the relevant `tests/eval_tools/` coverage. Timing, evaluation, or scoring changes
also require before/after evidence from at least one representative task when
compatible GPU hardware is available.

## Artifacts, destructive actions, and security

Treat `workspace_*`, `logs/`, report bundles, profiler output, and generated
held-out data as user-owned experiment artifacts. Do not run `make cleanup-works`,
delete them, or overwrite a run unless explicitly requested.
Do not commit generated workspaces, logs, cloned runtime dependencies,
credentials, authentication state, or evaluation-tool artifacts.

Privileged GPU containers and per-task workspaces are reproducibility boundaries,
not security sandboxes. Avoid exposing mounted credentials through prompts,
logs, generated code, commands, or reports. New mounts, downloads, subprocesses,
and external execution paths require an explicit security and reproducibility
review.
