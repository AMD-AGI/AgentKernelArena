---
myst:
    html_meta:
        "description": "Supported and tested hardware, Docker images, software versions, agent CLIs, and model providers for AgentKernelArena."
        "keywords": "AgentKernelArena, compatibility matrix, Docker, SGLang, ROCm, AMD Instinct, Python, PyTorch, GPU, agents, model providers"
---

# AgentKernelArena compatibility matrix

Use the following matrix to view AgentKernelArena compatibility and system requirements:

```{raw} html
<table>
  <thead>
    <tr>
      <th>AgentKernelArena version</th>
      <th>Category</th>
      <th>Component</th>
      <th>AMD Instinct GPU</th>
      <th>ROCm</th>
      <th>Python</th>
      <th>PyTorch</th>
      <th>Docker image</th>
      <th>Notes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="19">0.1.0</td>
      <td rowspan="2">Hardware</td>
      <td>GPU architecture</td>
      <td>MI300 series</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td><code>target_gpu_model: MI300</code></td>
    </tr>
    <tr>
      <td>GPU architecture</td>
      <td>MI355X</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
    </tr>
    <tr>
      <td rowspan="10">Software</td>
      <td>Docker</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>Current stable release. Required; serial evaluations run through <code>make docker-run</code>; multi-GPU evaluations run through <code>make docker-parallel-run</code>.</td>
    </tr>
    <tr>
      <td>SGLang benchmark image</td>
      <td>MI300 series (<code>gfx942</code>)</td>
      <td>7.2</td>
      <td>—</td>
      <td>—</td>
      <td><code>lmsysorg/sglang:v0.5.12-rocm720-mi30x</code></td>
      <td>Override with <code>AKA_DOCKER_IMAGE</code> or <code>AKA_DOCKER_IMAGE_GFX942</code>.</td>
    </tr>
    <tr>
      <td>SGLang benchmark image</td>
      <td>MI355X (<code>gfx950</code>)</td>
      <td>7.2</td>
      <td>—</td>
      <td>—</td>
      <td><code>lmsysorg/sglang-rocm:v0.5.14-rocm720-mi35x-20260705</code></td>
      <td>Verified digest: <code>sha256:b435b508b5aa696abb25c909341ce73e41574c4271cf716bed72418dcea86b78</code>. Override with <code>AKA_DOCKER_IMAGE</code> or <code>AKA_DOCKER_IMAGE_GFX950</code>.</td>
    </tr>
    <tr>
      <td>Python</td>
      <td>—</td>
      <td>—</td>
      <td>3.10</td>
      <td>—</td>
      <td>—</td>
      <td>Bundled in the SGLang image.</td>
    </tr>
    <tr>
      <td>PyTorch</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>ROCm build</td>
      <td>—</td>
      <td>Provided by the SGLang Docker image.</td>
    </tr>
    <tr>
      <td>Triton</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>Bundled with the image's ROCm PyTorch. Required for Triton task categories.</td>
    </tr>
    <tr>
      <td>AITER</td>
      <td>MI355X (<code>gfx950</code>)</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>Version <code>0.1.17.dev110+g9127c94a1</code> in the verified <code>gfx950</code> image. Required by AITER-backed task oracles and kernels.</td>
    </tr>
    <tr>
      <td>FlyDSL</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>Version <code>0.2.2</code> in the verified <code>gfx950</code> image, or run <code>make docker-setup-flydsl</code> when absent. Required for <code>flydsl2flydsl</code>, <code>torch2flydsl</code>, and <code>triton2flydsl</code> tasks.</td>
    </tr>
    <tr>
      <td>hipcc</td>
      <td>—</td>
      <td>Matches image ROCm</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>Required for HIP tasks.</td>
    </tr>
    <tr>
      <td>rocprof-compute</td>
      <td>—</td>
      <td>Matches image ROCm</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>Required for HIP performance profiling.</td>
    </tr>
    <tr>
      <td rowspan="3">Agents</td>
      <td>Cursor Agent CLI</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>See <a href="../install/install.md">Installation</a> for setup instructions.</td>
    </tr>
    <tr>
      <td>Claude Code</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>See <a href="../install/install.md">Installation</a> for setup instructions.</td>
    </tr>
    <tr>
      <td>Codex CLI</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>See <a href="../install/install.md">Installation</a> for setup instructions.</td>
    </tr>
    <tr>
      <td rowspan="4">Model providers</td>
      <td>OpenAI</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>Requires <code>OPENAI_API_KEY</code>.</td>
    </tr>
    <tr>
      <td>Anthropic</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>Requires <code>ANTHROPIC_API_KEY</code>.</td>
    </tr>
    <tr>
      <td>OpenRouter</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>Requires <code>OPENROUTER_API_KEY</code>.</td>
    </tr>
    <tr>
      <td>Local vLLM</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>Self-hosted on port <code>30001</code> using <code>make vllm</code>.</td>
    </tr>
  </tbody>
</table>
```
