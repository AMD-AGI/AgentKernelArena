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

| AMD Instinct GPU | ROCm version | Notes |
| --- | --- | --- |
| MI300X | 10.0.0 (Bundled in the selected SGLang image) | `target_gpu_model: MI300X` |
| MI325X | 10.0.0 (Bundled in the selected SGLang image) | `target_gpu_model: MI325X` |
| MI355X | 10.0.0 (Bundled in the selected SGLang image) | `target_gpu_model: MI355X` |

## Software requirements

The following software versions are required or verified.

| Component | Version | Notes |
| --- | --- | --- |
| Linux | Ubuntu 22.04, Ubuntu 24.04 | |
| hipcc | Matches ROCm image | Required for HIP tasks. |
| rocprof-compute | Matches ROCm image | Required for HIP performance profiling. |
| Docker | Current stable release | Required; serial experiments run through `make docker-run`; multi-GPU experiments run through `make docker-parallel-run`. |
| SGLang runtime image | `lmsysorg/sglang:v0.5.12-rocm720-mi30x` for `gfx942`; `lmsysorg/sglang-rocm:v0.5.14-rocm720-mi35x-20260705` for `gfx950` | The verified `gfx950` digest is `sha256:b435b508b5aa696abb25c909341ce73e41574c4271cf716bed72418dcea86b78`. Override with `AKA_DOCKER_IMAGE`, `AKA_DOCKER_IMAGE_GFX942`, or `AKA_DOCKER_IMAGE_GFX950`. |
| Python | Provided by the image (for example, 3.10) | Bundled in the SGLang image. |
| Node.js and npm | Node.js 22 with a current npm | Required on the host only for the alternative npm installation of Claude Code or another npm-installed agent CLI. |
| PyTorch | ROCm build bundled in the image | Provided by the SGLang Docker image. |
| Triton | Bundled with the image's ROCm PyTorch | Required for Triton task categories. |
| AITER | `0.1.17.dev110+g9127c94a1` in the verified `gfx950` image | Required by AITER-backed task oracles and kernels. |
| FlyDSL | `0.2.2` in the verified `gfx950` image (or `make docker-setup-flydsl` when absent) | Required for `flydsl2flydsl`, `torch2flydsl`, and `triton2flydsl` tasks. |

## Evaluation-tool sidecars

Optional Triton FpSan, GPU ASan, rocJITsu, and HIP-FpSan dependencies are kept
out of the scoring image and installed in one isolated sidecar image per tool.
The scoring image, FlyDSL, and AITER versions in the preceding table remain
unchanged.

| GPU architecture | Sidecar status | Notes |
| --- | --- | --- |
| `gfx950` (MI355X) | Runtime-qualified, candidate-dependent | Pinned image/build locks and all four integrated startup controls pass on the current hardware. End-to-end readiness still depends on language, artifact, adapter, and candidate attestation. Trusted single-dispatch Triton/FlyDSL rocJITsu capsule replay is implemented, but automatic evaluator-owned capsule capture and binding to the correctness run remain advisory-only gaps. |
| `gfx942` (MI300X/MI325X) | Unverified | No equivalent image/adapter/positive-control qualification has completed; the host runner currently rejects evaluation-tool sidecars. |

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
| `geak_v3` | GEAK CLI; HIP-oriented integration. |
| `geak_v3_triton` | GEAK CLI; Triton-oriented integration. |
| `mini_swe_triton` | mini-swe-agent/GEAK dependencies. |
| `task_validator` | Claude Code or Codex backend configured in `agents/task_validator/agent_config.yaml`. |

## Model providers

Model/provider support is integration-specific; run configuration files do not
configure a provider.

| Provider | Notes |
| --- | --- |
| OpenAI | Use a selected integration or CLI configured for OpenAI. |
| Anthropic | Use a selected integration or CLI configured for Anthropic. |
| OpenRouter or another OpenAI-compatible service | Supported when the selected integration accepts a custom provider/base URL. |
| Local vLLM | `make vllm` starts an OpenAI-compatible endpoint on port `30001`; configure the selected integration to use it. |
