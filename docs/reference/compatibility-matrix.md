---
myst:
    html_meta:
        "description": "Supported and tested hardware, Docker images, software versions, agent CLIs, and model providers for AgentKernelArena."
        "keywords": "AgentKernelArena, compatibility matrix, Docker, SGLang, ROCm, AMD Instinct, Python, PyTorch, GPU, agents, model providers"
---

# AgentKernelArena compatibility matrix

## Hardware

The following hardware configurations are supported and tested.

| Component | Supported | Notes |
| --- | --- | --- |
| GPU architecture | AMD Instinct™ MI300 series | `target_gpu_model: MI300` |
| GPU architecture | AMD Instinct™ MI355X | |

## Software

The following software versions are required or verified.

| Component | Version | Notes |
| --- | --- | --- |
| Docker | Current stable release | Required; serial evaluations run through `make docker-run`; multi-GPU evaluations run through `make docker-parallel-run`. |
| SGLang benchmark image | `lmsysorg/sglang:v0.5.12-rocm720-mi30x` for `gfx942`; `lmsysorg/sglang-rocm:v0.5.14-rocm720-mi35x-20260705` for `gfx950` | The verified `gfx950` digest is `sha256:b435b508b5aa696abb25c909341ce73e41574c4271cf716bed72418dcea86b78`. Override with `AKA_DOCKER_IMAGE`, `AKA_DOCKER_IMAGE_GFX942`, or `AKA_DOCKER_IMAGE_GFX950`. |
| ROCm | Bundled in the selected SGLang image | The default images are ROCm 7.2 based. |
| Python | Provided by the image (for example, 3.10) | Bundled in the SGLang image. |
| PyTorch | ROCm build bundled in the image | Provided by the SGLang Docker image. |
| Triton | Bundled with the image's ROCm PyTorch | Required for Triton task categories. |
| AITER | `0.1.17.dev110+g9127c94a1` in the verified `gfx950` image | Required by AITER-backed task oracles and kernels. |
| FlyDSL | `0.2.2` in the verified `gfx950` image (or `make docker-setup-flydsl` when absent) | Required for `flydsl2flydsl`, `torch2flydsl`, and `triton2flydsl` tasks. |
| hipcc | Matches image ROCm | Required for HIP tasks. |
| rocprof-compute | Matches image ROCm | Required for HIP performance profiling. |

## Agents

The following agent CLIs are supported with AgentKernelArena. See
[Installation](../install/install.md) for setup instructions.

- Cursor Agent CLI
- Claude Code
- Codex CLI

## Model providers

The following model providers are supported.

| Provider | Notes |
| --- | --- |
| OpenAI | Requires `OPENAI_API_KEY`. |
| Anthropic | Requires `ANTHROPIC_API_KEY`. |
| OpenRouter | Requires `OPENROUTER_API_KEY`. |
| Local vLLM | Self-hosted on port `30001` using `make vllm`. |
