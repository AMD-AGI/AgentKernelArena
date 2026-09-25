---
myst:
    html_meta:
        "description": "Supported and tested hardware, Docker images, software versions, agent CLIs, and model providers for AgentKernelArena."
        "keywords": "AgentKernelArena, compatibility matrix, Docker, SGLang, ROCm, AMD Instinct, Python, PyTorch, GPU, agents, model providers"
---

# AgentKernelArena compatibility matrix

Supported and tested hardware, Docker images, software versions, agent CLIs, and model providers for AgentKernelArena.

## Hardware requirements

The following hardware configurations are supported and tested.

| AMD GPU | ROCm version | Notes |
| --- | --- | --- |
| MI300X | 7.2 (Bundled in the selected SGLang image.) | `target_gpu_model: MI300X` |
| MI325X | 7.2 (Bundled in the selected SGLang image.) | `target_gpu_model: MI325X` |
| MI355X | 7.2 (Bundled in the selected SGLang image.) | `target_gpu_model: MI355X` |
| RDNA4 (`gfx1201`, 16 GB tested) | Pinned in the [RDNA4 recipe](../../docker/rdna4/Dockerfile) | `target_gpu_model: RDNA4`; HIP/Triton task runtime checks, with limits in the [recipe guide](../../docker/rdna4/README.md). |

## Software requirements

The following software versions are required or verified.

| Component | Version | Notes |
| --- | --- | --- |
| Linux | Ubuntu 22.04, Ubuntu 24.04 | |
| hipcc | Matches ROCm image | Required for HIP tasks. |
| Profiler tools | Match runtime image | Smoke reports `rocprof-compute` and `rocprofv3` availability. Core graph/event timing needs neither; profiling runs can require specific binaries with `AKA_REQUIRED_PROFILERS`. Availability does not establish candidate analysis. See the [qualification record](runtime-upgrade-qualification.md#profiler-capability-policy). |
| Docker | Current stable release | Required; serial experiments run through `make docker-run`; multi-GPU experiments run through `make docker-parallel-run`. |
| SGLang runtime image | `lmsysorg/sglang:v0.5.12-rocm720-mi30x` for `gfx942`; `lmsysorg/sglang-rocm:v0.5.14-rocm720-mi35x-20260705` for `gfx950` | The verified `gfx950` digest is `sha256:b435b508b5aa696abb25c909341ce73e41574c4271cf716bed72418dcea86b78`. Override with `AKA_DOCKER_IMAGE`, `AKA_DOCKER_IMAGE_GFX942`, or `AKA_DOCKER_IMAGE_GFX950`. |
| RDNA4 runtime image | [Digest-pinned base and layout adapter](../../docker/rdna4/Dockerfile) | Default image builds on first use if missing; `make docker-build-rdna4` prebuilds or rebuilds it. Image overrides disable automatic builds; see the [runtime guide](../../docker/rdna4/README.md). |
| Python for GPU experiments | Provided by the qualified task runtime image | Task-specific qualification applies; older images do not support every retained task. |
| Python for the full repository CPU/source audit | CPython 3.12 | The complete source set requires 3.12 syntax, and migration AST fingerprints are pinned to this minor version. Use full Git history; see [contributor verification](../../CONTRIBUTING.md#testing-and-verification). |
| Node.js and npm | Node.js 22+ for Claude Code's npm installation; Node.js 24 for DeepSeek Harness | Required on the host for npm-installed agent CLIs; DeepSeek uses a dedicated prefix. |
| PyTorch | ROCm build bundled in the image | Provided by the selected runtime image. |
| Triton | Bundled with the image's ROCm PyTorch | Required for Triton task categories. |
| AITER | `0.1.17.dev110+g9127c94a1` in the verified `gfx950` image | Required by AITER-backed task oracles and kernels. |
| FlyDSL | `0.2.2` in the verified `gfx950` image (or `make docker-setup-flydsl` when absent) | Required for `flydsl2flydsl`, `torch2flydsl`, `triton2flydsl`, and `operator2flydsl` tasks. |

## Evaluation-tool sidecars

Optional Triton FpSan, GPU ASan, rocJITsu Race Detector, rocJITsu Waitcheck,
rocJITsu ConSan, and HIP-FpSan dependencies are kept out of the scoring image
and installed in one isolated sidecar image per tool. The scoring image,
FlyDSL, and AITER versions in the preceding table remain unchanged.

| GPU architecture | Sidecar status | Notes |
| --- | --- | --- |
| `gfx950` (MI355X) | Runtime-qualified, candidate-dependent | Pinned image/build locks and all six integrated startup controls pass on the current hardware. End-to-end readiness still depends on language, artifact, adapter, and candidate attestation. Waitcheck and ConSan are qualified only for explicitly configured advisory pilots. Trusted single-dispatch Triton/FlyDSL rocJITsu capsule replay is implemented, but automatic evaluator-owned capsule capture and binding to the correctness run remain advisory-only gaps. |
| `gfx942` (MI300X/MI325X) | Unverified | No equivalent image/adapter/positive-control qualification has completed; the host runner currently rejects evaluation-tool sidecars. |
| `gfx1201` (RDNA4) | Unverified | Runtime task checks do not qualify evaluation-tool sidecars; the host runner rejects them. |

The runtime base digest and per-tool package/source locks are recorded in
`docker/eval-tools/images.lock.yaml`. See [Check kernels with evaluation
tools](../how-to/use-evaluation-tools.md#strict-support-matrix) for the strict
Triton, HIP, FlyDSL, AITER, rocBLAS, and RCCL matrix. Normal task compatibility
does not imply sanitizer coverage. Tool startup resolves both the selected
scoring-image reference and the pinned `gfx950` SGLang content-addressed
manifest reference to immutable local image IDs and requires those local IDs to
match. Aliases of that exact image are allowed, but rebuilt, upgraded, or
retagged images are rejected. The scoring container is launched by the verified
image ID.

## Agents

The following templates are selectable in the current `AgentType` registry. See
[Install AgentKernelArena](../install/install.md) and
[Configure agents and models](../how-to/agents.md) for setup instructions.

| Template | Runtime dependency |
| --- | --- |
| `cursor` | Cursor Agent CLI and host login state. |
| `claude_code` | Native/local or npm-installed Claude Code CLI and host login state. |
| `codex` | Codex CLI and host login state. |
| `deepseek_harness` | Dedicated Node.js 24 prefix with the `dsh` version pinned in [agent_config.yaml](../../agents/deepseek_harness/agent_config.yaml), plus `DEEPSEEK_API_KEY`; see [setup](../../agents/deepseek_harness/README.md). |
| `forge` | Pinned KernelForge engine and selected backend/provider; see the [Forge guide](../../agents/forge/README.md). |
| `geak` | Pinned GEAK checkout, Claude Code Workflow support and SDK; see the [GEAK guide](../../agents/geak/README.md). |
| `geak_v4` | Registry alias for `geak`; uses the same v2 launcher, configuration and Workflow runtime. |
| `forge_operator2flydsl` | Compatibility name for `forge`, using the same launcher and configuration. |
| `task_validator` | Claude Code or Codex backend, with defaults in `agents/task_validator/agent_config.yaml` and run-level `agent` overrides. |

Selectable names are defined in [the registry](../../src/module_registration.py).
This dependency list does not assert that every agent/model/task combination
has completed qualification.

DeepSeek Harness qualification is recorded in
[PR #111](https://github.com/AMD-AGI/AgentKernelArena/pull/111), including the
schema-v2 SIKL FlyDSL GEMM and HIP quantization image tasks on MI355X through
Slurm and Docker. Results apply to the recorded task, candidate, and runtime;
earlier HIP GELU and Triton RMSNorm observations predate the schema-v2 rebase.
Its MI300 example, RDNA4 execution, and multi-GPU scheduling have not yet been
validated on hardware.

## Model providers

Model/provider support and run-level overrides are integration-specific; there
is no shared top-level provider field.

| Provider | Notes |
| --- | --- |
| OpenAI | Use a selected integration or CLI configured for OpenAI. |
| Anthropic | Use a selected integration or CLI configured for Anthropic. |
| DeepSeek | `deepseek_harness` selects the upstream DeepSeek provider; model, protocol, and optional endpoint override are integration-local settings. |
| OpenRouter or another OpenAI-compatible service | Supported when the selected integration accepts a custom provider/base URL. |
| Local vLLM | `make vllm` uses a separate serving image to launch an OpenAI-compatible endpoint on port `30001`; configure the selected integration to use it. This serving path is unverified on RDNA4; the [RDNA4 kernel-runtime checks](../../docker/rdna4/README.md#validation-and-limits) do not establish full vLLM serving support. |
