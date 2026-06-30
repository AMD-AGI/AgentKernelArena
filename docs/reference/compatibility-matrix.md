---
myst:
    html_meta:
        "description": "Supported and tested hardware, Docker images, software versions, agent CLIs, and model providers for AgentKernelArena."
        "keywords": "AgentKernelArena, compatibility matrix, Docker, SGLang, ROCm, AMD Instinct, Python, PyTorch, GPU, agents, model providers"
---

# AgentKernelArena compatibility matrix

The following versions are the supported and tested configurations for
AgentKernelArena. Entries marked `TODO (verify)` must be confirmed against a
tested environment before publication.

## Hardware

The following hardware configurations are supported and tested.

| Component | Supported | Notes |
| --- | --- | --- |
| GPU architecture | AMD Instinct™ MI300 series | `target_gpu_model: MI300`. TODO (verify) other architectures. |
| GPU architecture | AMD Instinct MI355X | TODO (verify) |

## Software

The following software versions are required or verified.

| Component | Version | Notes |
| --- | --- | --- |
| Docker | Current stable release | Required; evaluations run through `make docker-run`. |
| Benchmark image | `lmsysorg/sglang:v0.5.12-rocm720-mi30x` for `gfx942`; `lmsysorg/sglang:v0.5.12-rocm720-mi35x` for `gfx950`; `rocm/vllm-dev:rocm7.2.1_navi_ubuntu24.04_py3.12_pytorch_2.9_vllm_0.16.0` for Navi/RDNA/Radeon `gfx10*`/`gfx11*`/`gfx12*` | Override with `AKA_DOCKER_IMAGE`, `AKA_DOCKER_IMAGE_GFX942`, `AKA_DOCKER_IMAGE_GFX950`, `AKA_DOCKER_IMAGE_NAVI`, or `AKA_DOCKER_IMAGE_<ARCH>`. |
| ROCm | Bundled in the selected benchmark image | The default images are ROCm 7.2 based. |
| Python | Provided by the image | Bundled in the selected benchmark image. |
| PyTorch | ROCm build bundled in the image | Provided by the selected benchmark image. |
| Triton | Bundled with the image's ROCm PyTorch | Required for Triton task categories. |
| FlyDSL | Provided by the image (or `make docker-setup-flydsl`) | Required for `flydsl2flydsl` tasks. |
| hipcc | Matches image ROCm | Required for HIP tasks. |
| rocprof-compute | Matches image ROCm | Required for HIP performance profiling. |

## Agents

The following agent CLIs have been tested with AgentKernelArena.

| Agent | Tested version | Notes |
| --- | --- | --- |
| Cursor Agent CLI | TODO (verify) | Installed using `make install-cursor-agent`. |
| Claude Code | TODO (verify) | `npm install -g @anthropic-ai/claude-code`. |
| Codex CLI | TODO (verify) | Installed per the official Codex CLI instructions. |

## Model providers

The following model providers are supported.

| Provider | Notes |
| --- | --- |
| OpenAI | Requires `OPENAI_API_KEY`. |
| Anthropic | Requires `ANTHROPIC_API_KEY`. |
| OpenRouter | Requires `OPENROUTER_API_KEY`. |
| Local vLLM | Self-hosted on port `30001` using `make vllm`. |
