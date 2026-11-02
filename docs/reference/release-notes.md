---
myst:
    html_meta:
        "description": "Release notes for AgentKernelArena, including current A/B experimentation, RL-ready signals, GPU task environments, and known limitations."
        "keywords": "AgentKernelArena, release notes, A/B testing, agent RL, ROCm, GPU kernel, HIP, Triton, agents"
---

# AgentKernelArena release notes

This topic summarizes the features available in each AgentKernelArena release. For the hardware and software versions validated for a release, see the [Compatibility matrix](compatibility-matrix.md).

## AgentKernelArena 0.4.0

AgentKernelArena 0.4.0 makes Arena an RL-ready environment with bounded,
independently evaluated rewards. Every task now uses one schema-v2 contract.
The release adds fixed-workload end-to-end model tasks and production-operator
suites, and repairs task correctness and benchmark integrity across the suite.

### Release highlights

#### Rewards and evaluation

- Replaced the additive point score with a bounded reward in `[0, 1]`:
  `max(0, 1 - 1.05 / speedup)`. Compilation and correctness are prerequisites
  with no point bonuses, and a correct candidate at or below 1.05× scores zero.
  An interim square-root curve used on `main` during development was replaced
  by this linear default (`SCORE_POWER` 1.0) before the release. The linear
  curve has a finite slope at the threshold, so scores just above it do not
  swing with timing noise. See [Scoring](api-reference.md#scoring).
- Reported agent execution and candidate source changes separately from
  independent candidate acceptance in results, summaries, CSV export and run
  comparison.
- Added an opt-in wall-clock `budget_mode: time` for Claude Code, Cursor, Codex
  and DeepSeek Harness, and per-task `agent_costs/` records with
  provider-reported cost and token usage. Unreported values are `null`, not
  estimated. See [Agents and models](../how-to/agents.md).

#### Unified task contract and new task scopes

- Migrated all retained tasks to one TaskSpec v2 `config.yaml` contract. It
  separates the provided baseline from the candidate, supports initially
  unimplemented candidates, and keeps workloads, numerical comparison and
  evaluation task-owned. Every integration uses the same candidate and
  evaluator interfaces. See
  [Task definition, schema, and authoring](../how-to/add-task.md).
- Added end-to-end tasks for model-wide kernel implementation and fusion under
  fixed weights, precision, parallelism, serving flags and workloads.
  Evaluation uses paired baseline/candidate serving runs and a separate
  read-only candidate verifier. The 50 Magpie recipes (45 text, three
  multimodal and two AgentX) have individual selectable task directories and
  shared runtime binding. Catalog coverage does not imply that every
  model/runtime combination has passed GPU qualification. See
  [End-to-end tasks](../how-to/e2e-kernel-tasks.md).
- Added `sikl_task_builder` (`make docker-sikl-task-builder`), which converts
  SIKL bundles into self-contained schema-v2 tasks while preserving their
  workloads, references and comparison rules. See
  [Task builder](../how-to/sikl-task-builder.md).
- Added 46 production-operator tasks under `Aiter-task`: BF16 and block-scaled
  FP8 GEMM, MXFP4 MoE, MLA, mHC and paged top-k, including the DeepSeek-V4
  Flash operator set. Added three MI300X FlyDSL BF16 GEMM variants under
  `Aiter-task-mi300`.
- Added split-K GEMM, top-k routing, large paged attention, A8W4 MoE and SGLang
  request-to-token pool tasks.
- Version 0.4.0 ships 472 task packages, including 52 end-to-end tasks.
  Availability depends on each task's runtime and architecture contract.

#### Benchmark integrity and task quality

- Repaired 114 GPU task packages for correctness and benchmark integrity:
  narrower editable boundaries, independently drawn timed inputs with replay
  checks, restored input state and wider correctness coverage. Each changed
  task's README describes its revision. The protected wrappers of the
  symbol-scoped Triton tasks fix the launch configuration (tiles, `num_warps`
  and `num_stages`).
- Strengthened correctness and replay checks for further Triton, HIP and
  FlyDSL tasks, added rotating timed inputs for AITER GEMM tasks, and protected
  evaluator state in the FlyDSL task runtimes.
- Added the canonical cold-cache graph protocol, used by the new MI300X GEMM,
  A8W4 MoE and request-to-token pool tasks. It prepares inputs, runs a
  read-only 1 GiB eviction sweep, then times one operator graph replay with
  same-stream GPU Events. An opt-in `cuda_graph_cold_weights` method is also
  available; no task uses it yet. See
  [Performance measurement methodology](benchmark-methodology.md).
- The harness guard now rejects added runtime patching and library operator
  delegation before scoring.
- Added bounded `repair_only` task maintenance with fresh validator results,
  preservation of scored cases and numerical gates, and independent review of
  retained findings. Repairs are not optimization gains. See
  [Quality loop](../how-to/quality-loop.md).

#### Agents and runtime

- Added the opt-in `deepseek_harness` agent with a pinned headless CLI,
  API-key authentication, isolated per-task sessions, and Docker/Slurm
  quickstarts. See the [integration guide](https://github.com/AMD-AGI/AgentKernelArena/blob/main/agents/deepseek_harness/README.md)
  for installation, model settings, and validation limits. `AGENTS=all`
  continues to check Cursor, Claude Code, and Codex.
- Provisioned the pinned Forge engine in Docker (`make docker-setup-forge`),
  with a separate installation per engine revision for concurrent campaigns.
- Updated the MI355X default to the immutable SGLang 0.5.20 / ROCm 10 runtime.
  Scoring and resumed workspaces are bound to their runtime image.
  Task-specific runtime requirements and optional analysis-tool qualifications
  remain separate; see [MI355X runtime and compatibility](mi355x-runtime.md).
- Added shared CDNA 3 (`gfx942`) architecture guidance with separate MI300X,
  MI325X and MI300A hardware profiles and quickstarts. This adds
  model-specific configuration and prompt coverage; physical GPU qualification
  remains task- and runtime-specific.
- Added `configuration_symbols`, which makes the launch settings of
  symbol-scoped Triton tasks tunable while their host wrappers stay protected.
  It is piloted on `triton2triton/vllm/triton_bmm`.
- Reorganized the README and guides around task scopes, a minimal Docker run
  and reward interpretation.

### Upgrade notes

- Scores are not comparable with 0.3.0. The 0.3.0 score was `120 + 100 ×
  speedup` points for a correct candidate; the 0.4.0 score is bounded in
  `[0, 1]`, and 2× now scores 0.475. Recompute both arms rather than
  relabeling historical results.
- Some repaired tasks change their timed input distributions or invocation
  preparation, so recompute their baselines. Compare scores only within one
  task revision.
- HIP task directories are flat (`hip2hip/gpumode/GELU` is now
  `hip2hip/GELU`). Duplicate `instruction2triton` tasks were removed and the
  unique flip task moved to `triton2triton/rocmbench`. The `repository` tasks
  were removed.
- Retired five ROCmBench primitive tasks (`test_reverse_range`,
  `test_triton_swizzle2d`, `test_random_int`, `test_randn`, `test_kernel_sub`)
  and 16 `torch2hip/gpumode` generation tasks whose PyTorch references and
  workload manifests match retained `hip2hip` optimization tasks. Remove these
  retired selectors from custom run configs; use the corresponding HIP
  optimization task when appropriate. Earlier qualification records describe
  their original revisions.
- Custom tasks must use the schema-v2 contract; see
  [Task definition, schema, and authoring](../how-to/add-task.md).
- To reproduce an older MI355X result, select `GFX950_V0514_IMMUTABLE_IMAGE`
  with `docker_image` or `AKA_DOCKER_IMAGE` and start a fresh run. Evaluation
  tools need the matching repository revision; the ROCm 10 tool profile
  rejects the old scoring image.
- Generic `MI300` no longer assumes MI300X topology or memory limits. Select
  `MI300X`, `MI325X` or `MI300A` for a model-specific profile; the `MI325`
  alias remains supported.
- Removed the standalone `make vllm` local model-serving target and its
  provider documentation.
- Reports from earlier revisions lack the agent-execution and candidate-change
  fields; treat them as unknown.
- Removed retired-task-only CPU tests and parameter cases. CPU CI uses Python
  3.12, limits nested numerical-library threads and uploads per-test timing
  results.

### Known limitations

- End-to-end tasks need a prepared runtime binding, GPU groups and evaluation
  budget. Whole-model tasks currently support the Codex, Claude Code, Cursor
  and DeepSeek Harness integrations; GEAK and KernelForge adaptation is
  separate work.
- Numerical checks and semantic review have finite coverage and do not prove
  the absence of every invalid implementation.
- Task additions and repairs carry GPU qualification evidence scoped to the
  runtime and source revision recorded with them. CPU checks do not establish
  GPU correctness or performance for every task.
- AgentKernelArena supplies the environment and reward signals; it does not
  include an RL trainer, replay buffer, or policy-update loop.
- Evaluation-tool sidecars are experimental and `gfx950`-only. Startup controls
  prove a tool installation can detect its synthetic bug, not that a candidate
  was instrumented. No bundled task currently supplies a production-qualified
  adapter/attestation. All six integrated startup controls passed on the
  ROCm 7.2 MI355X qualification host; the ROCm 10 profile has separate
  [qualification requirements](runtime-upgrade-qualification.md). Synthetic manager-to-sidecar candidate pairs also
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

## AgentKernelArena 0.3.0

AgentKernelArena 0.3.0 expands kernel-agent integrations, adds task quality
auditing and optional GPU analysis tools, and strengthens benchmark integrity
across the task suite.

### Release highlights

#### Agents and task quality

- Added GEAK v4 integration, including lifecycle handling for its background
  optimization workflow. Arena independently checks and scores the resulting
  kernel.
- Added Forge integration and compatibility with KernelForge as shipped in
  Hyperloom, including kernel identity metadata, editable-source boundaries,
  and task-owned correctness and benchmark drivers.
- Added the separate `quality_loop` workflow: task validation and bounded
  repair, one optimization candidate, centralized evaluation, independent
  review, and gated baseline/case improvements.
- Strengthened task-validator report finalization, completion handling, and
  benchmark-fairness checks.

#### Benchmark integrity and task coverage

- Expanded canonical graph-replay timing across kernel benchmarks, with
  device-event fallback and state restoration support. Added shared Python and
  native HIP helpers, stricter case matching, and benchmark-method consistency
  checks.
- Added harness integrity checks around agent execution and strengthened
  protection of test/configuration files, performance entrypoints, and
  colocated harness code.
- Added the `image_kernel` task type for kernels shipped in the selected
  runtime image, plus MI355X attention, MoE, quantization, and related tasks.
- Added FlyDSL optimization examples grouped by compute-pattern complexity.
- Version 0.3.0 shipped 426 task packages across nine task types. This
  historical count is not the current task inventory.

#### Runtime and optional analysis

- Added a Slurm/Spur wrapper that allocates one or eight MI355X GPUs before launching Docker on the assigned compute node.
- Added native standalone Codex mounting alongside the existing npm installation path.
- Added a pinned, non-root RDNA4 `gfx1201` Docker runtime, automatic first-use
  image builds, an RDNA4 quickstart, and architecture-specific agent guidance.
- Added opt-in, per-tool sidecar plumbing and typed reports for Triton FpSan,
  ROCm GPU ASan, rocJITsu Race Detector, rocJITsu Waitcheck, rocJITsu ConSan,
  and HIP-FpSan. The initial sidecar locks are verified only for `gfx950`.
  Workers automatically execute synthetic startup controls, while useful
  candidate runs still require task-specific adapters and attestations.

### Upgrade notes

- Removed the unused `geak_v3`, `geak_v3_triton`, and `mini_swe_triton`
  integrations, their configuration files, and their shared parallel-count
  helper. These names, including hyphenated and uppercase spellings, are no
  longer accepted as `agent.template` values. New runs must select a
  [supported agent](../how-to/agents.md#supported-agents); to resume an old run
  with its original retired agent, use a revision from before this removal.
- Removed the retired integrations' `num_parallel` and `run_mode` settings
  from the configuration reference. GEAK v4 and Forge remain supported, and
  GEAK-derived task packages and saved experiment results are unchanged.
- `quality_loop` is a separate repository maintenance workflow. Normal task
  runs still use centralized evaluation; they do not automatically start its
  independent reviewer.

### Known limitations

- Evaluation tools were experimental, candidate-dependent and qualified only
  on MI355X (`gfx950`). A passing sidecar startup control did not establish
  that a candidate was instrumented. See the current
  [evaluation-tool guide](../how-to/use-evaluation-tools.md).
- RDNA4 runtime checks were performed on a 16 GB `gfx1201` device; task
  memory, LDS, language, and architecture restrictions still apply.

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
- Improved environment handling for PyTorch, Triton, MIOpen, and HIP tasks.
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

Version 0.2.0 shipped 397 task packages. This historical count is not the current task inventory.

The legacy 184-task `instruction2triton/tritonbench` suite and several obsolete HIP tasks were removed as part of repository cleanup.

#### More reliable performance measurement

- Added CUDA-graph timing with automatic CUDA-event fallback.
- Records the timing method with benchmark results to make mixed-method comparisons visible.
- Moved benchmark ownership out of agent-editable kernels for the remaining GEAK tasks.
- Added canonical shared performance helpers under `src/tools/perf/`.
- Added CI checks to keep task-local performance-helper stubs synchronized.
- Strengthened handling of warmup, repeated measurements, per-shape results, and baseline-versus-optimized comparisons.

#### Agent and validator updates

The agent templates shipped in 0.2.0 were:

- `claude_code`
- `codex`
- `cursor`
- `geak_v3`
- `geak_v3_triton`
- `mini_swe_triton`
- `task_validator`

The task validator now includes Codex backend support, improved Python-environment propagation, stronger source and target checks, starter-stub detection, and standardized validation reports.

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
- Added `/usr/bin/time` to the container where required by build scripts.
- Rejected missing or unimplemented generated targets before performance scoring.
- Improved benchmark integrity by moving correctness and timing logic outside editable kernel files.

### Upgrade notes

- Docker is now required for supported experiment execution.
- The root `requirements.txt` and host-venv workflow have been removed.
- The legacy `SWE_agent`, `geak_hip`, `geak_optimagentv2`, `geak_ourllm_kernel2kernel`, `openevolve`, and `single_llm_call` templates were removed. The GEAK v3 and mini-swe replacements shipped in this release have since been retired; see [0.3.0](#agentkernelarena-030) for their removal.
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
task directories included `hip2hip`, `triton2triton`, `instruction2triton`,
`torch2hip`, and `flydsl2flydsl`. Later development replaced
several agent integrations and added the current parallel runner, shared
performance helpers, FlyDSL conversion suites, and other capabilities listed
above.
