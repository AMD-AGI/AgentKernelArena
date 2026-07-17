# Contributing to AgentKernelArena

Thanks for your interest in AgentKernelArena! This guide explains how to contribute, report issues, and submit changes.

## Before You Start

- Read `README.md` to understand the project scope: controlled A/B experiments and RL-ready feedback for GPU kernel agents.
- Skim the files under `example_configs/` for run-level agent/task/GPU selection and the relevant `agents/<name>/agent_config.yaml` for agent-specific model and runtime settings.
- Ensure you have an AMD GPU with ROCm-compatible Docker access; the supported workflow uses the pinned ROCm/SGLang images documented in the compatibility matrix.
- Confirm that the selected agent integration and its authentication/dependencies are available.

## Development Setup

Docker is the only supported path. All runs happen inside the pinned ROCm/SGLang
container; see `docs/install/install.md`.

```bash
# Verify the container can see Python, ROCm tools, and the GPU
make docker-smoke

# Select a run config and verify only its agent
CONFIG_PATH=example_configs/quickstart_claude_mi300.yaml
make docker-check-agents CONFIG="$CONFIG_PATH"

# Optional strict check of all three first-class CLIs
make docker-check-agents AGENTS=all

# Optional: install the Cursor Agent CLI on the host (so it can be mounted)
make install-cursor-agent

# Optional: install FlyDSL when the image lacks it (for all three FlyDSL task types)
make docker-setup-flydsl

# Optional: install local commit hooks
pre-commit install

# Optional: start a local OpenAI-compatible vLLM endpoint. Connecting an agent
# to it is integration-specific; the endpoint does not reconfigure agents.
make vllm
```

## Workflow

1. Create a new branch from `main`.
2. Keep changes focused and scoped.
3. Run a smoke test against at least one task before submitting:

```bash
make docker-run CONFIG=example_configs/quickstart_claude_mi300.yaml
```

4. Open a Pull Request with motivation, impact, and verification steps.

## Code Style and Quality

- Follow PEP 8 for Python code.
- Keep agent integrations isolated under `agents/<agent_name>/` — don't leak agent-specific logic into `src/`.
- Update the relevant example run configurations, docs, `AgentType`, and the launcher/handler branches in `src/module_registration.py` when adding a new agent. `agents/__init__.py` only provides the shared decorator registry.
- Add documentation or comments when intent is non-obvious.
- Performance timing helpers are generated into run workspaces from
  `src/tools/perf/`.
  Do not hand-edit `tasks/*/rocmbench/**/performance_utils_pytest.py` stubs or the
  `AKA-GENERATED` block in vLLM `task_runner.py` files. Edit `src/tools/perf/`
  instead, and run `make check-perf-helpers` before pushing.
  Use `make materialize-perf-workspace WORKSPACE=...` or
  `make materialize-perf-task TASK=tasks/...` when you need a local copy with
  the real helper code injected.

## Testing and Verification

This project depends on GPU hardware/drivers and orchestrates external LLM agent CLIs. In your PR, include:

- Test environment (GPU model, ROCm version, Docker image, OS)
- Agent(s) used and their versions
- Task selector exercised (for example `hip2hip`, `triton2triton`, `instruction2triton`, `torch2hip`, a FlyDSL task type, or `repository`)
- Key commands and output summary, e.g.:

```bash
make docker-run CONFIG=example_configs/quickstart_claude_mi300.yaml
python3 compare_runs.py <run-directory-1> <run-directory-2>
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
- Extend task coverage across HIP, Triton, FlyDSL, PyTorch conversion, instruction-generated, or repository-level tasks
- Improve scoring or fairness logic in `src/score.py`
- Improve A/B comparison, experiment tracking, or visualization (`src/visualization/`)
- Add support for new models / providers (OpenAI, Anthropic, OpenRouter, vLLM)
- Improve docs, examples, and tests

## License

By contributing, you agree that your contributions are licensed under the repository `LICENSE` (Apache License 2.0).
