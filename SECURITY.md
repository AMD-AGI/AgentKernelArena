# Security Policy

## Reporting a Vulnerability

**Do not open a public GitHub issue.** Report privately via one of:

- **GitHub Private Vulnerability Reporting:** [Report a vulnerability](https://github.com/AMD-AGI/AgentKernelArena/security/advisories/new)
- **AMD Product Security portal:** https://www.amd.com/en/resources/product-security.html

Please include: description and impact, steps to reproduce, and affected versions or commits.

We aim to acknowledge reports within 1 business day.

## Scope

This policy covers code and configuration in this repository — the AgentKernelArena orchestration framework (`main.py`, `src/`), agent integrations under `agents/`, task definitions under `tasks/`, and workspace isolation logic.

Because AgentKernelArena launches third-party LLM agent CLIs (Cursor Agent, Claude Code, Codex, SWE-agent, GEAK, etc.) inside per-task workspaces and executes the kernel code those agents produce, please flag any of the following privately:

- Sandbox escape from a per-task workspace
- Credential leakage through prompts, logs, or generated code
- Supply-chain risk from dynamically installed agent CLIs or model SDKs
- Resource-exhaustion or denial-of-service against the host runner

For issues in the upstream agent CLIs themselves (Cursor, Claude Code, Codex, SWE-agent, GEAK) or in model providers (OpenAI, Anthropic, OpenRouter, vLLM), report to those vendors directly.

For AMD product issues unrelated to this repo, use the [AMD Product Security portal](https://www.amd.com/en/resources/product-security.html).
