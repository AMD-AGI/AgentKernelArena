---
myst:
    html_meta:
        "description": "Release notes for AgentKernelArena, including current A/B experimentation, RL-ready signals, GPU task environments, and known limitations."
        "keywords": "AgentKernelArena, release notes, A/B testing, agent RL, ROCm, GPU kernel, HIP, Triton, agents"
---

# AgentKernelArena release notes

This topic summarizes the features available in each AgentKernelArena release. For the hardware and software versions validated for a release, see the [Compatibility matrix](compatibility-matrix.md).

## AgentKernelArena 0.2.0

AgentKernelArena 0.2.0 evolves the initial kernel-agent framework into a
Docker-first platform for controlled A/B experiments, scalable multi-GPU
execution, and RL-ready GPU kernel evaluation.

### Release highlights

#### Controlled experimentation and evaluation

- Added first-class A/B experimentation workflows with labeled baseline and treatment runs.
- Exposed compilation, correctness, latency, speedup, and score fields as structured signals for external agent-RL systems.
- Added run comparison through `src/tools/compare_runs.py` and the standalone visualization dashboard.
- Added held-out evaluation for testing kernel generalization on unseen shapes.
- Centralized compilation, correctness, performance measurement, result generation, and scoring outside agent-editable code.

#### Docker-first execution

- Docker is now the supported execution path; the legacy host virtual-environment workflow has been removed.
- Added architecture-aware ROCm/SGLang runtime selection for gfx942 and gfx950.
- Added GPU, agent CLI, authentication-state, and writable runtime-cache provisioning.
- Improved environment handling for PyTorch, Triton, MIOpen, HIP, and repository-level tasks.
- Stopped mounting host SSH credentials into benchmark containers.

#### Multi-GPU parallel runs

- Added `make docker-parallel-run`.
- Runs one long-lived worker container per GPU.
- Workers atomically claim tasks from a shared run-local queue.
- Added per-worker GPU visibility, HOME, cache, and agent-state isolation.
- Runs aggregation and post-processing once after all workers finish.
- Preserved the existing serial `make docker-run` workflow.

#### Expanded task coverage

This release adds 146 task packages:

- 33 KernelBench-derived torch2hip tasks across Levels 1–3.
- 45 torch2flydsl tasks.
- 51 triton2flydsl tasks.
- 17 GEAK-oriented triton2triton tasks covering GEMM, attention, MoE, normalization, quantization, routing, and other workloads.

Version 0.2.0 contains 397 task packages across `hip2hip`, `instruction2triton`, `torch2hip`, `torch2flydsl`, `triton2triton`, `triton2flydsl`, `flydsl2flydsl`, and `repository`.

The legacy 184-task `instruction2triton/tritonbench` suite and several obsolete HIP tasks were removed as part of repository cleanup.

#### More reliable performance measurement

- Added CUDA-graph timing with automatic CUDA-event fallback.
- Records the timing method with benchmark results to make mixed-method comparisons visible.
- Moved benchmark ownership out of agent-editable kernels for the remaining GEAK tasks.
- Added canonical shared performance helpers under `src/tools/perf/`.
- Added CI checks to keep task-local performance-helper stubs synchronized.
- Strengthened handling of warmup, repeated measurements, per-shape results, and baseline-versus-optimized comparisons.

#### Agent and validator updates

The supported agent templates are now:

- `claude_code`
- `codex`
- `cursor`
- `geak_v3`
- `geak_v3_triton`
- `mini_swe_triton`
- `task_validator`

The task validator now includes Codex backend support, repository-task validation, improved Python-environment propagation, stronger source and target checks, starter-stub detection, and standardized validation reports.

#### Documentation and onboarding

- Reorganized the documentation around installation, experimentation, agents, task authoring, validation, parallel execution, held-out evaluation, visualization, and benchmark methodology.
- Added MI300/MI300X and MI355X quickstarts.
- Added a curated 60-task MI355X Cursor benchmark configuration.
- Improved setup guidance for native and npm-installed Claude Code, Codex, and Cursor Agent.
- Clarified that task workspaces provide reproducibility and separation between runs, but are not security sandboxes.

### Notable fixes

- Corrected a GELU implementation that previously computed ReLU.
- Fixed large-shape reduction accuracy in `InnerProd` and `MaskedLanguageModel`.
- Strengthened `ball_query` correctness validation against its CPU reference.
- Fixed MIOpen cache permission and lockfile failures.
- Ensured repository task subprocesses use the ROCm-enabled Python environment.
- Added `/usr/bin/time` to the container where required by build scripts.
- Rejected missing or unimplemented generated targets before performance scoring.
- Improved benchmark integrity by moving correctness and timing logic outside editable kernel files.

### Upgrade notes

- Docker is now required for supported experiment execution.
- The root `requirements.txt` and host-venv workflow have been removed.
- The legacy `SWE_agent`, `geak_hip`, `geak_optimagentv2`, `geak_ourllm_kernel2kernel`, `openevolve`, and `single_llm_call` templates were removed. Use `geak_v3`, `geak_v3_triton`, or `mini_swe_triton` for current GEAK-oriented workflows.
- The legacy instruction2triton/tritonbench task paths are no longer available.
- Held-out evaluation moved under `src.held_out`.
- Visualization is now invoked through `python3 -m src.visualization`.
- The former root run configuration moved to `example_configs/benchmark_cursor_mi355x.yaml`.
- `make docker-run` now defaults to the MI300 Claude Code quickstart.

### Known limitations

- AgentKernelArena provides RL-ready environments and reward signals, but does not include an RL trainer, replay buffer, or policy-update loop.
- One agent template is selected per run; heterogeneous agents must be compared through separate labeled runs.
- `cuda2hip` is recognized by the prompt system, but no bundled cuda2hip task suite is currently included.
- Local vLLM provider configuration remains specific to the selected agent integration.
- GPU task execution requires compatible physical AMD hardware and ROCm driver access.

## AgentKernelArena 0.1.0

The initial release established the core task-discovery, workspace,
agent-launch, evaluation, scoring, logging, and report-generation pipeline.

At that release, the registry included Cursor, Claude Code, Codex, SWE-agent,
single-call, OpenEvolve, and earlier GEAK integrations. The bundled top-level
task directories were `hip2hip`, `triton2triton`, `instruction2triton`,
`torch2hip`, `flydsl2flydsl`, and `repository`. Later development replaced
several agent integrations and added the current parallel runner, shared
performance helpers, FlyDSL conversion suites, and other capabilities listed
above.
