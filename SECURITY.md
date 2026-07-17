# Security Policy

## Reporting a Vulnerability

**Do not open a public GitHub issue.** Report privately via one of:

- **GitHub Private Vulnerability Reporting:** [Report a vulnerability](https://github.com/AMD-AGI/AgentKernelArena/security/advisories/new)
- **AMD Product Security portal:** https://www.amd.com/en/resources/product-security.html

Please include: description and impact, steps to reproduce, and affected versions or commits.

We aim to acknowledge reports within 1 business day.

## Scope

This policy covers code and configuration in this repository — the AgentKernelArena orchestration framework (`main.py`, `src/`), agent integrations under `agents/`, task definitions under `tasks/`, and workspace isolation logic.

AgentKernelArena launches third-party LLM agent CLIs (Cursor Agent, Claude Code,
Codex, and specialized integrations) and executes the kernel code they produce.
Per-task workspaces separate experiment artifacts for reproducibility and
concurrency; they are not security sandboxes. The Docker runner uses privileged
GPU containers and mounts repository and agent-authentication state. Please flag
any of the following privately:

- Unexpected access outside the assigned task workspace or to mounted credentials/repository data
- Credential leakage through prompts, logs, or generated code
- Supply-chain risk from dynamically installed agent CLIs or model SDKs
- Resource-exhaustion or denial-of-service against the host runner

For issues in the upstream agent CLIs themselves (Cursor, Claude Code, Codex) or in model providers (OpenAI, Anthropic, OpenRouter, vLLM), report to those vendors directly.

For AMD product issues unrelated to this repo, use the [AMD Product Security portal](https://www.amd.com/en/resources/product-security.html).
