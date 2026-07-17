---
myst:
    html_meta:
        "description": "Release notes for AgentKernelArena, including current A/B experimentation, RL-ready signals, GPU task environments, and known limitations."
        "keywords": "AgentKernelArena, release notes, A/B testing, agent RL, ROCm, GPU kernel, HIP, Triton, agents"
---

# AgentKernelArena release notes

## Current main (unreleased)

The current development branch positions AgentKernelArena as a controlled A/B
experimentation platform and RL-ready task environment for GPU kernel agents.

### Current capabilities

- **A/B experimentation**: run a fixed task set with baseline and treatment
  agent configurations, label both runs, and compare their reports.
- **RL-ready signals**: expose compilation, correctness, timing, speedup, and
  score fields for external agent-RL systems.
- **Current agent templates**: `cursor`, `claude_code`, `codex`, `geak_v3`,
  `geak_v3_triton`, `mini_swe_triton`, and `task_validator`.
- **Task environments**: `hip2hip`, `cuda2hip`, `triton2triton`,
  `instruction2triton`, `torch2hip`, `torch2flydsl`, `triton2flydsl`,
  `flydsl2flydsl`, and `repository`. `cuda2hip` is recognized by the prompt
  system but does not yet have a bundled task suite.
- **Docker-first execution**: select pinned ROCm/SGLang images from the target
  GPU architecture.
- **Example run configurations**: provide one-task Claude Code quickstarts for
  MI300/MI300X and MI355X plus a curated 60-task Cursor benchmark for MI355X
  under `example_configs/`.
- **Centralized outcomes**: independently measure compilation, correctness, and
  GPU performance, then compute a configurable score.
- **Visualization module**: build and serve the comparison dashboard through
  `python3 -m src.visualization`, with generated data kept outside `src/`.
- **Timing provenance**: record baseline and optimized timing methods so
  mixed-method comparisons are visible.
- **Workspace reproducibility**: preserve each task in a timestamped workspace.
- **Performance-helper materialization**: maintain shared Triton timing helpers
  in `src/tools/perf/` and inject them into run workspaces.
- **Resumable and multi-GPU runs**: skip completed tasks and schedule remaining
  work through one Docker worker per GPU.
- **Task validator**: run 10 task-quality checks and emit structured validation
  reports.
- **Held-out evaluation**: generate unseen shapes and measure kernel
  generalization through the optional `src.held_out` module.
- **Local run comparison**: compare run reports with `compare_runs.py` or the
  visualization dashboard.

### Known limitations

- One agent template is selected per run; heterogeneous agents are compared
  through separate labeled runs rather than one mixed worker queue.
- AgentKernelArena provides environments and reward signals but not an RL
  trainer, replay buffer, or policy-update loop.
- A single agent task can run until its configured timeout. During a parallel
  run, other workers continue processing the remaining queue.
- `make vllm` starts a local endpoint, but provider wiring is specific to the
  selected agent integration.
- Several task suites are still being expanded.

## AgentKernelArena 0.1.0 (2026-06-18)

The initial release established the core task-discovery, workspace,
agent-launch, evaluation, scoring, logging, and report-generation pipeline.

At that release, the registry included Cursor, Claude Code, Codex, SWE-agent,
single-call, OpenEvolve, and earlier GEAK integrations. The bundled top-level
task directories were `hip2hip`, `triton2triton`, `instruction2triton`,
`torch2hip`, `flydsl2flydsl`, and `repository`. Later development replaced
several agent integrations and added the current parallel runner, shared
performance helpers, FlyDSL conversion suites, and other capabilities listed
above.
