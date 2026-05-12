# Contributing to AgentKernelArena

Thanks for your interest in AgentKernelArena! This guide explains how to contribute, report issues, and submit changes.

## Before You Start

- Read `README.md` to understand the project scope — a standardized arena for evaluating LLM coding agents on GPU kernel optimization tasks.
- Skim `config.yaml` to understand how agents, tasks, and LLM parameters are configured.
- Ensure you have a supported GPU environment (AMD GPU with ROCm 6.4+ / 7.0+ / 7.1+ — the Makefile auto-detects).
- Confirm you have access to at least one supported agent (Cursor Agent, Claude Code, Codex, SWE-agent, GEAK, or single-LLM-call) and any required API keys.

## Development Setup

```bash
# Complete environment setup (auto-detects ROCm, creates .venv, installs deps)
make setup-venv

# Activate the venv
make act

# Optional: install Cursor Agent CLI
make install-cursor-agent

# Optional: start a local vLLM server for self-hosted models
make vllm
```

## Workflow

1. Create a new branch from `main`.
2. Keep changes focused and scoped.
3. Run a smoke test against at least one task before submitting:

```bash
python main.py        # uses config.yaml
```

4. Open a Pull Request with motivation, impact, and verification steps.

## Code Style and Quality

- Follow PEP 8 for Python code.
- Keep agent integrations isolated under `agents/<agent_name>/` — don't leak agent-specific logic into `src/`.
- Update `config.yaml`, `README.md`, and the agent registry (`agents/__init__.py`) when adding a new agent.
- Add documentation or comments when intent is non-obvious.

## Testing and Verification

This project depends on GPU hardware/drivers and orchestrates external LLM agent CLIs. In your PR, include:

- Test environment (GPU model, ROCm version, Python 3.12, OS)
- Agent(s) used and their versions
- Task category exercised (rocm-examples, rocprim, customer_hip, triton, torch2hip)
- Key commands and output summary, e.g.:

```bash
python main.py
python compare_runs.py --runs <run-id-1> <run-id-2>
```

- For changes to scoring or evaluation logic, attach before/after results on at least one task category.

## Filing Issues

Please include:

- Reproduction steps (exact `config.yaml` snippet or command flags)
- Expected vs actual behavior
- Environment (OS, GPU, ROCm version, Python version, agent CLI version)
- Relevant logs from the workspace under `works/` or a minimal repro

## Security

If you discover a security issue, do not open a public issue. Contact maintainers through a private channel.

This project executes third-party AI agents in isolated workspaces — flag any sandbox-escape, credential-leakage, or supply-chain concerns privately.

## Suggested Contributions

- Add new agent integrations under `agents/`
- Extend task coverage (new HIP, Triton, or Torch2HIP tasks under `tasks/`)
- Improve scoring or fairness logic in `src/score.py`
- Improve the leaderboard or visualization (`visualization/`)
- Add support for new models / providers (OpenAI, Anthropic, OpenRouter, vLLM)
- Improve docs, examples, and tests

## License

By contributing, you agree that your contributions are licensed under the repository `LICENSE` (MIT).
