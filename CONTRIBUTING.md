# Contributing to AgentKernelArena

Thanks for your interest in AgentKernelArena! This guide explains how to contribute, report issues, and submit changes.

## Before You Start

- Read `README.md` to understand the project scope: controlled A/B experiments and RL-ready feedback for GPU kernel agents.
- Before adding or modifying a task, read [Task definition, schema, and authoring](docs/how-to/add-task.md), It is the single authoring reference for schema-v3 kernel tasks. Every new or changed task needs a framework-finalized `qa_report.yaml` with `status: PASS` from the [QA auditor](docs/how-to/qa-auditor.md) on MI355X before its PR.
- Skim the files under `example_configs/` for run-level agent/task/GPU selection and the relevant `agents/<name>/agent_config.yaml` for agent-specific model and runtime settings.
- Ensure you have an AMD GPU with ROCm-compatible Docker access; use the architecture-specific runtime documented in the compatibility matrix.
- Confirm that the selected agent integration and its authentication/dependencies are available.

## Development Setup

Docker is the only supported path. All runs happen inside the selected GPU
runtime container; see `docs/install/install.md` for image selection and the
RDNA4 image preparation.

```bash
# Verify the container can see Python, ROCm tools, and the GPU
make docker-smoke

# Select a run config and verify only its agent
CONFIG_PATH=example_configs/quickstart_claude_mi300.yaml
make docker-check-agents CONFIG="$CONFIG_PATH"

# Optional strict check of Cursor, Claude Code, and Codex
make docker-check-agents AGENTS=all

# Optional: install the Cursor Agent CLI on the host (so it can be mounted)
make install-cursor-agent

# Optional: install FlyDSL when the image lacks it (for FlyDSL tasks)
make docker-setup-flydsl

# Optional: install local commit hooks
pre-commit install
```

DeepSeek Harness is opt-in: follow its [setup guide](agents/deepseek_harness/README.md)
and select its run config or use `AGENTS=deepseek_harness`. `AGENTS=all` retains
the three-CLI check above.

## Workflow

1. Create a new branch from `main`.
2. Keep changes focused and scoped.
3. Run a smoke test against at least one task before submitting:

```bash
make docker-run CONFIG=example_configs/quickstart_claude_mi300.yaml
```

4. Complete the repository cleanliness checks below before committing.
5. Open a Pull Request with motivation, impact, and verification steps.

## Repository cleanliness before committing

Keep source, tests, required task inputs, reusable configurations, and maintained
documentation in the repository. Keep temporary plans, scratch scripts, PR task
lists, retry configs, status notes, and raw QA reports outside it unless
their inclusion is explicitly requested. `example_configs/` is for maintained
run examples, not a record of individual audit campaigns.

When Docker or Slurm needs a local config in the mounted checkout, use an
ignored `config_*.yaml` file, such as `config_experiment.yaml`. Verify that it
is ignored and untracked; do not force-add it. Keep generated outputs in the
configured run directories. Task packages must not ship previous `build/`
output, `qa_report*`, `.qa_complete`, `task_result.*`, or `qa_summary.json`.
Intentional test fixtures and required input/reference data remain versioned.

Before each commit:

```bash
git status --short --untracked-files=all
git diff --stat
git diff --cached --name-status
git diff --cached --check
git diff --cached
```

Stage explicit paths and review each added file for a lasting purpose. Check
references before deleting or moving a file; a PR number or an old date alone
does not make a regression test or qualification record disposable. Adding an
ignore rule does not remove an already tracked artifact from the commit.

Summarize validation commands, outcomes, and limitations in the PR. Retain the
underlying reports, source/config identity, and runtime details in an external
artifact location, and link the relevant evidence when it can be shared. Before
removing tracked reports, preserve a recoverable copy outside the repository.
Remove only disposable scratch files created for the current change; preserve
pre-existing user files, workspaces, logs, and experiment results.

## Code Style and Quality

- Follow PEP 8 for Python code.
- Keep agent integrations isolated under `agents/<agent_name>/` — don't leak agent-specific logic into `src/`.
- Update the relevant example run configurations, docs, `AgentType`, and the launcher/handler branches in `src/module_registration.py` when adding a new agent. `agents/__init__.py` only provides the shared decorator registry.
- Add documentation or comments when intent is non-obvious.
- Performance timing helpers are generated into run workspaces from
  `src/tools/perf/`.
  Do not commit generated copies (`_aka_benchmark.py`, `_aka_serving*.py` or
  `_aka_eval/`) under `tasks/`. Edit `src/tools/perf/` instead, and run
  `make check-perf-helpers` before pushing. The check
  (`src/tools/sync_perf_helpers.py`) fails if a generated helper is committed or
  if a schema-v2 (serving) task is neither a serving recipe nor a serving client;
  it audits task sources only and is not GPU qualification.

## Testing and Verification

Use **CPython 3.12** for the full CPU/mock suite and repository-wide source
compilation, matching `.github/workflows/perf-helpers.yml`. Some task helpers use
Python 3.12 f-string syntax, and the committed migration AST fingerprints were
recorded with 3.12; those fingerprints are not portable across Python minor
versions. Python 3.11 is not a supported interpreter for the full repository
audit. Use a full Git checkout (`git fetch --unshallow` for an existing shallow
clone), because preservation tests read historical task sources with `git show`.
Keep those source comparisons enabled.

For CPU/mock checks, set `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` and
`OPENBLAS_NUM_THREADS=1`, matching CI. Many reference and negative-control
tests perform small tensor operations; large nested thread pools add overhead.
CI uploads JUnit test timings so slow cases can be investigated individually.

This audit requirement does not upgrade the Docker scoring images. Actual GPU
runs still use the task's qualified image and Python version; an older image's
task-specific PASS does not establish compatibility with every retained task.
See the [compatibility matrix](docs/reference/compatibility-matrix.md).

This project depends on GPU hardware/drivers and orchestrates external LLM agent CLIs. In your PR, include:

- Test environment (GPU model, ROCm version, Docker image, OS)
- Agent(s) used and their versions
- Task selector and `target_language` exercised (for example `normalization` with `triton`, or a `qa_auditor` run), and the `qa_report.yaml` status for new or changed tasks
- Key commands and output summary, e.g.:

```bash
make docker-run CONFIG=example_configs/quickstart_claude_mi300.yaml
python3 src/tools/compare_runs.py <run-directory-1> <run-directory-2>
```

- For changes to scoring or evaluation logic, attach before/after results on at least one task category.
- For changes to `src/tools/perf/`, also include `make check-perf-helpers` output.

## Filing Issues

Please include:

- Reproduction steps (exact run-configuration snippet or command flags)
- Expected vs actual behavior
- Environment (OS, GPU, ROCm version, Python version, agent CLI version)
- Relevant files from `logs/` and `workspace_<gpu>_<agent>/run_<timestamp>/`, or a minimal repro

## Security

If you discover a security issue, do not open a public issue. Contact maintainers through a private channel.

This project executes third-party AI agents permissively inside privileged Docker containers. Per-task workspaces are a reproducibility boundary, not a security sandbox; report unexpected access to mounted credentials, repository files, or host resources privately.

## Suggested Contributions

- Add new agent integrations under `agents/`
- Extend schema-v3 task coverage across operators and production scenarios, and harden language support (Triton, Gluon, HIP, FlyDSL) beyond MI355X
- Improve scoring or fairness logic in `src/score.py`
- Improve A/B comparison, experiment tracking, or visualization (`src/visualization/`)
- Improve support for additional model providers and local serving backends
- Improve docs, examples, and tests

## License

By contributing, you agree that your contributions are licensed under the repository `LICENSE` (Apache License 2.0).
