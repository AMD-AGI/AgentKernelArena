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
- Added opt-in, per-tool sidecar plumbing and typed reports for Triton FpSan,
  ROCm GPU ASan, rocJITsu Race Detector, rocJITsu Waitcheck, rocJITsu ConSan,
  and HIP-FpSan. The initial sidecar locks are verified only for `gfx950`.
  Workers automatically execute synthetic startup controls, while useful
  candidate runs still require task-specific adapters and attestations.
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
- Evaluation-tool sidecars are experimental and `gfx950`-only. Startup controls
  prove a tool installation can detect its synthetic bug, not that a candidate
  was instrumented. No bundled task currently supplies a production-qualified
  adapter/attestation. All six integrated startup controls pass on the current
  MI355X qualification host. Synthetic manager-to-sidecar candidate pairs also
  distinguished clean from seeded-bug Triton FpSan, HIP/Triton GPU ASan, and
  HIP-FpSan runs; trusted AOT replay produced a clean Triton result and found the
  seeded FlyDSL LDS race. Waitcheck distinguished a correct wait from a missing
  `lgkmcnt(0)`, and strict ConSan record/replay distinguished clean single-wave
  LDS accesses from seeded cross-wave conflicts. These fixtures do not qualify
  a bundled task. Keep the policy advisory until each selected candidate path
  is independently qualified.
- Evaluation-tool parsing and execution now fail closed on ambiguous FpSan
  markers, process failure, non-finite protocol numbers, stale artifact reuse,
  unsupported GPU-library kernels, and replay capsules without a golden output.
  Each attempt receives a fresh artifact directory. The worker contains the
  complete descendant process tree, including session-detached children, and
  treats required cleanup as a failed execution. Replay launch geometry is
  validated as exact `uint32` input and checked against runtime device limits.
- Native HIP rocJITsu no longer accepts task-launcher stdout/stderr as clean
  dispatch attestation; only canonical records in the evaluator-owned report
  sink are eligible. Waitcheck and ConSan startup controls now execute their
  production entrypoints and parsers. The Waitcheck control covers inventory
  and the C API, while ConSan uses exact raw HSACO identities, separate
  instrumented/oracle argv, and an oracle that rejects leaked hook state.
- The current runner exposes evaluation-tool sockets and agent-writable report
  paths during the same container run. Per-tool writable socket directories and
  a read-only top-level artifact namespace plus one writable
  `.eval-tool-artifacts/<label>` child prevent cross-socket mutation,
  cross-worker writable aliases, and the previous writable whole-`experiments`
  alias. Sibling reports remain visible read-only. The current worker report
  directory is explicitly mounted read/write into scoring, including when the
  quality-loop repository root is read-only. The agent can still call
  unauthenticated evaluator sockets and write its own worker's diagnostic
  artifacts in the same phase. Tool reports are not a tamper-resistant reward
  boundary until agent and evaluation phases are separated and candidate
  provenance is strengthened.
- Tool startup requires the selected scoring image to resolve to the same
  immutable local Docker image ID as the pinned SGLang content-addressed
  manifest reference. The selected reference and ID are serialized in plan
  evidence. Build-attestation artifact paths are relative to and contained
  below the attestation directory; absolute and escaping paths are rejected.
- Triton/FlyDSL rocJITsu AOT replay now validates a workspace-contained,
  language-matched, single-dispatch `gfx950` capsule, forbids task launchers, and
  invokes the image-owned native replay helper. Automatic evaluator-owned
  capsule capture and binding to the ordinary correctness dispatch are not yet
  implemented, so this path remains advisory.

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
