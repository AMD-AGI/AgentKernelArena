---
myst:
    html_meta:
        "description": "Supported and tested hardware, Docker images, software versions, agent CLIs, and model providers for AgentKernelArena."
        "keywords": "AgentKernelArena, compatibility matrix, Docker, SGLang, ROCm, AMD Instinct, Python, PyTorch, GPU, agents, model providers"
---

# AgentKernelArena compatibility matrix

Use the following matrix to view AgentKernelArena compatibility and system requirements:

| Category | Component | AMD Instinct GPU | ROCm | Python | PyTorch | Docker image | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Hardware | GPU architecture | MI300 series | — | — | — | — | `target_gpu_model: MI300` |
| Hardware | GPU architecture | MI355X | — | — | — | — | |
| Software | Docker | — | — | — | — | — | Current stable release. Required; serial evaluations run through `make docker-run`; multi-GPU evaluations run through `make docker-parallel-run`. |
| Software | SGLang benchmark image | MI300 series (`gfx942`) | 7.2 | — | — | `lmsysorg/sglang:v0.5.12-rocm720-mi30x` | Override with `AKA_DOCKER_IMAGE` or `AKA_DOCKER_IMAGE_GFX942`. |
| Software | SGLang benchmark image | MI355X (`gfx950`) | 7.2 | — | — | `lmsysorg/sglang-rocm:v0.5.14-rocm720-mi35x-20260705` | Verified digest: `sha256:b435b508b5aa696abb25c909341ce73e41574c4271cf716bed72418dcea86b78`. Override with `AKA_DOCKER_IMAGE` or `AKA_DOCKER_IMAGE_GFX950`. |
| Software | Python | — | — | 3.10 | — | — | Bundled in the SGLang image. |
| Software | PyTorch | — | — | — | ROCm build | — | Provided by the SGLang Docker image. |
| Software | Triton | — | — | — | — | — | Bundled with the image's ROCm PyTorch. Required for Triton task categories. |
| Software | AITER | MI355X (`gfx950`) | — | — | — | — | Version `0.1.17.dev110+g9127c94a1` in the verified `gfx950` image. Required by AITER-backed task oracles and kernels. |
| Software | FlyDSL | — | — | — | — | — | Version `0.2.2` in the verified `gfx950` image, or run `make docker-setup-flydsl` when absent. Required for `flydsl2flydsl`, `torch2flydsl`, and `triton2flydsl` tasks. |
| Software | hipcc | — | Matches image ROCm | — | — | — | Required for HIP tasks. |
| Software | rocprof-compute | — | Matches image ROCm | — | — | — | Required for HIP performance profiling. |
| Agents | Cursor Agent CLI | — | — | — | — | — | See [Installation](../install/install.md) for setup instructions. |
| Agents | Claude Code | — | — | — | — | — | See [Installation](../install/install.md) for setup instructions. |
| Agents | Codex CLI | — | — | — | — | — | See [Installation](../install/install.md) for setup instructions. |
| Model providers | OpenAI | — | — | — | — | — | Requires `OPENAI_API_KEY`. |
| Model providers | Anthropic | — | — | — | — | — | Requires `ANTHROPIC_API_KEY`. |
| Model providers | OpenRouter | — | — | — | — | — | Requires `OPENROUTER_API_KEY`. |
| Model providers | Local vLLM | — | — | — | — | — | Self-hosted on port `30001` using `make vllm`. |
