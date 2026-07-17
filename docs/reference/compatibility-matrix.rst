.. meta::
   :description: Supported and tested hardware, Docker images, software versions, agent CLIs, and model providers for AgentKernelArena.
   :keywords: AgentKernelArena, compatibility matrix, Docker, SGLang, ROCm, AMD Instinct, Python, PyTorch, GPU, agents, model providers

.. _aka-compat-matrix:

*******************************************
AgentKernelArena compatibility matrix
*******************************************

Use the following matrix to view AgentKernelArena compatibility and system
requirements:

.. list-table::
   :widths: 10 12 22 16 8 10 12 24
   :header-rows: 1
   :align: left
   :class: compat-matrix format-big-table

   * - AKA version
     - Category
     - Component
     - AMD Instinct GPU
     - ROCm
     - Python
     - PyTorch
     - Docker image
   * - 0.1.0
     - Hardware
     - GPU architecture
     - MI300 series (``gfx942``) :sup:`1`
     - —
     - —
     - —
     - —
   * -
     -
     - GPU architecture
     - MI355X (``gfx950``) :sup:`2`
     - —
     - —
     - —
     - —
   * -
     - Software
     - Docker :sup:`3`
     - —
     - —
     - —
     - —
     - —
   * -
     -
     - SGLang runtime image
     - MI300 series (``gfx942``)
     - 7.2
     - 3.10
     - ROCm build
     - ``lmsysorg/sglang:v0.5.12-rocm720-mi30x`` :sup:`4`
   * -
     -
     - SGLang runtime image
     - MI355X (``gfx950``)
     - 7.2
     - 3.10
     - ROCm build
     - ``lmsysorg/sglang-rocm:v0.5.14-rocm720-mi35x-20260705`` :sup:`5`
   * -
     -
     - Node.js and npm :sup:`6`
     - —
     - —
     - —
     - —
     - —
   * -
     -
     - Triton :sup:`7`
     - —
     - —
     - —
     - —
     - —
   * -
     -
     - AITER :sup:`8`
     - MI355X (``gfx950``)
     - —
     - —
     - —
     - —
   * -
     -
     - FlyDSL :sup:`9`
     - —
     - —
     - —
     - —
     - —
   * -
     -
     - hipcc :sup:`10`
     - —
     - —
     - —
     - —
     - —
   * -
     -
     - rocprof-compute :sup:`11`
     - —
     - —
     - —
     - —
     - —
   * -
     - Agents
     - First-class CLI templates :sup:`12`
     - —
     - —
     - —
     - —
     - —
   * -
     -
     - Specialized templates :sup:`13`
     - —
     - —
     - —
     - —
     - —
   * -
     -
     - Validation template :sup:`14`
     - —
     - —
     - —
     - —
     - —
   * -
     - Model providers
     - OpenAI :sup:`15`
     - —
     - —
     - —
     - —
     - —
   * -
     -
     - Anthropic :sup:`15`
     - —
     - —
     - —
     - —
     - —
   * -
     -
     - OpenRouter :sup:`15`
     - —
     - —
     - —
     - —
     - —
   * -
     -
     - Local vLLM :sup:`16`
     - —
     - —
     - —
     - —
     - —

Notes
=====

Hardware
--------

- :sup:`1` **MI300 series**: Use ``target_gpu_model: MI300``.
- :sup:`2` **MI355X**: Use ``target_gpu_model: MI355X``.

Software
--------

- :sup:`3` **Docker**: Use the current stable release. Docker is required;
  serial experiments run through ``make docker-run`` and multi-GPU experiments
  run through ``make docker-parallel-run``.
- :sup:`4` **SGLang runtime image (gfx942)**: Override with
  ``AKA_DOCKER_IMAGE`` or ``AKA_DOCKER_IMAGE_GFX942``.
- :sup:`5` **SGLang runtime image (gfx950)**: The verified digest is
  ``sha256:b435b508b5aa696abb25c909341ce73e41574c4271cf716bed72418dcea86b78``.
  Override with ``AKA_DOCKER_IMAGE`` or ``AKA_DOCKER_IMAGE_GFX950``.
- :sup:`6` **Node.js and npm**: Node.js 22+ and a current npm are required on
  the host only for the alternative npm installation of Claude Code or another
  npm-installed agent CLI.
- :sup:`7` **Triton**: Bundled with the image's ROCm PyTorch. Required for
  Triton task categories.
- :sup:`8` **AITER**: Version ``0.1.17.dev110+g9127c94a1`` in the verified
  ``gfx950`` image. Required by AITER-backed task oracles and kernels.
- :sup:`9` **FlyDSL**: Version ``0.2.2`` in the verified ``gfx950`` image, or
  run ``make docker-setup-flydsl`` when absent. Required for
  ``flydsl2flydsl``, ``torch2flydsl``, and ``triton2flydsl`` tasks.
- :sup:`10` **hipcc**: Matches the image ROCm version. Required for HIP tasks.
- :sup:`11` **rocprof-compute**: Matches the image ROCm version. Required for
  HIP performance profiling.

Agents
------

- :sup:`12` **First-class CLI templates**: ``cursor`` uses Cursor Agent CLI,
  ``claude_code`` uses a native/local or npm-installed Claude Code CLI, and
  ``codex`` uses Codex CLI. Each integration reuses its supported host login
  state.
- :sup:`13` **Specialized templates**: ``geak_v3`` is HIP-oriented,
  ``geak_v3_triton`` is Triton-oriented, and ``mini_swe_triton`` uses
  mini-swe-agent/GEAK dependencies.
- :sup:`14` **Validation template**: ``task_validator`` uses the Claude Code or
  Codex backend configured in ``agents/task_validator/agent_config.yaml``.

See :doc:`Installation <../install/install>` and
:doc:`Configure agents and models <../how-to/agents>` for setup instructions.

Model providers
---------------

Model/provider support is integration-specific; run configuration files do not
configure a provider.

- :sup:`15` **OpenAI, Anthropic, and OpenRouter**: Use a selected integration
  or CLI configured for that provider. OpenRouter and other OpenAI-compatible
  services are supported when the integration accepts a custom provider or
  base URL.
- :sup:`16` **Local vLLM**: ``make vllm`` starts an OpenAI-compatible endpoint
  on port ``30001``; configure the selected integration to use it.
