.. meta::
   :description: Supported and tested hardware, Docker images, software versions, agent CLIs, and model providers for AgentKernelArena.
   :keywords: AgentKernelArena, compatibility matrix, Docker, SGLang, ROCm, AMD Instinct, Python, PyTorch, GPU, agents, model providers

.. _aka-compat-matrix:

*******************************************
AgentKernelArena compatibility matrix
*******************************************

Use the following matrix to view AgentKernelArena compatibility and system requirements:

.. table::
   :widths: 10 10 20 14 6 8 10 16
   :align: left
   :class: compat-matrix format-big-table

   +----------+------------------+------------------------------+------------------------+------+----------+------------+-------------------------------------------+
   | AKA      | Category         | Component                    | AMD Instinct GPU       | ROCm | Python   | PyTorch    | Docker image                              |
   | version  |                  |                              |                        |      |          |            |                                           |
   +==========+==================+==============================+========================+======+==========+============+===========================================+
   | 0.1.0    | Hardware         | GPU architecture             | MI300 series :sup:`1`  | —    | —        | —          | —                                         |
   +          +                  +                              +------------------------+------+----------+------------+-------------------------------------------+
   |          |                  |                              | MI355X                 | —    | —        | —          | —                                         |
   +          +------------------+------------------------------+------------------------+------+----------+------------+-------------------------------------------+
   |          | Software         | Docker :sup:`2`              | —                      | —    | —        | —          | —                                         |
   +          +                  +------------------------------+------------------------+------+----------+------------+-------------------------------------------+
   |          |                  | SGLang benchmark image       | MI300 series           | 7.2  | —        | —          | ``lmsysorg/sglang:``                      |
   |          |                  |                              | (``gfx942``) :sup:`3`  |      |          |            | ``v0.5.12-rocm720-mi30x``                 |
   +          +                  +                              +------------------------+------+----------+------------+-------------------------------------------+
   |          |                  |                              | MI355X                 | 7.2  | —        | —          | ``lmsysorg/sglang-rocm:``                 |
   |          |                  |                              | (``gfx950``) :sup:`4`  |      |          |            | ``v0.5.14-rocm720-mi35x-20260705``        |
   +          +                  +------------------------------+------------------------+------+----------+------------+-------------------------------------------+
   |          |                  | Python :sup:`5`              | —                      | —    | 3.10     | —          | —                                         |
   +          +                  +------------------------------+------------------------+------+----------+------------+-------------------------------------------+
   |          |                  | PyTorch :sup:`6`             | —                      | —    | —        | ROCm build | —                                         |
   +          +                  +------------------------------+------------------------+------+----------+------------+-------------------------------------------+
   |          |                  | Triton :sup:`7`              | —                      | —    | —        | —          | —                                         |
   +          +                  +------------------------------+------------------------+------+----------+------------+-------------------------------------------+
   |          |                  | AITER :sup:`8`               | MI355X (``gfx950``)    | —    | —        | —          | —                                         |
   +          +                  +------------------------------+------------------------+------+----------+------------+-------------------------------------------+
   |          |                  | FlyDSL :sup:`9`              | —                      | —    | —        | —          | —                                         |
   +          +                  +------------------------------+------------------------+------+----------+------------+-------------------------------------------+
   |          |                  | hipcc :sup:`10`              | —                      | —    | —        | —          | —                                         |
   +          +                  +------------------------------+------------------------+------+----------+------------+-------------------------------------------+
   |          |                  | rocprof-compute :sup:`11`    | —                      | —    | —        | —          | —                                         |
   +          +------------------+------------------------------+------------------------+------+----------+------------+-------------------------------------------+
   |          | Agents           | Cursor Agent CLI :sup:`12`   | —                      | —    | —        | —          | —                                         |
   +          +                  +------------------------------+------------------------+------+----------+------------+-------------------------------------------+
   |          |                  | Claude Code :sup:`12`        | —                      | —    | —        | —          | —                                         |
   +          +                  +------------------------------+------------------------+------+----------+------------+-------------------------------------------+
   |          |                  | Codex CLI :sup:`12`          | —                      | —    | —        | —          | —                                         |
   +          +------------------+------------------------------+------------------------+------+----------+------------+-------------------------------------------+
   |          | Model providers  | OpenAI :sup:`13`             | —                      | —    | —        | —          | —                                         |
   +          +                  +------------------------------+------------------------+------+----------+------------+-------------------------------------------+
   |          |                  | Anthropic :sup:`14`          | —                      | —    | —        | —          | —                                         |
   +          +                  +------------------------------+------------------------+------+----------+------------+-------------------------------------------+
   |          |                  | OpenRouter :sup:`15`         | —                      | —    | —        | —          | —                                         |
   +          +                  +------------------------------+------------------------+------+----------+------------+-------------------------------------------+
   |          |                  | Local vLLM :sup:`16`         | —                      | —    | —        | —          | —                                         |
   +----------+------------------+------------------------------+------------------------+------+----------+------------+-------------------------------------------+

Notes
=====

Hardware
--------

- :sup:`1` **GPU architecture** (MI300 series): ``target_gpu_model: MI300``

Software
--------

- :sup:`2` **Docker**: Current stable release. Required; serial evaluations run through ``make docker-run``; multi-GPU evaluations run through ``make docker-parallel-run``.
- :sup:`3` **SGLang benchmark image** (gfx942): Override with ``AKA_DOCKER_IMAGE`` or ``AKA_DOCKER_IMAGE_GFX942``.
- :sup:`4` **SGLang benchmark image** (gfx950): Verified digest: ``sha256:b435b508b5aa696abb25c909341ce73e41574c4271cf716bed72418dcea86b78``. Override with ``AKA_DOCKER_IMAGE`` or ``AKA_DOCKER_IMAGE_GFX950``.
- :sup:`5` **Python**: Bundled in the SGLang image.
- :sup:`6` **PyTorch**: Provided by the SGLang Docker image.
- :sup:`7` **Triton**: Bundled with the image's ROCm PyTorch. Required for Triton task categories.
- :sup:`8` **AITER**: Version ``0.1.17.dev110+g9127c94a1`` in the verified ``gfx950`` image. Required by AITER-backed task oracles and kernels.
- :sup:`9` **FlyDSL**: Version ``0.2.2`` in the verified ``gfx950`` image, or run ``make docker-setup-flydsl`` when absent. Required for ``flydsl2flydsl``, ``torch2flydsl``, and ``triton2flydsl`` tasks.
- :sup:`10` **hipcc**: Matches image ROCm. Required for HIP tasks.
- :sup:`11` **rocprof-compute**: Matches image ROCm. Required for HIP performance profiling.

Agents
------

- :sup:`12` **Cursor Agent CLI**, **Claude Code**, **Codex CLI**: See :doc:`Installation <../install/install>` for setup instructions.

Model providers
---------------

- :sup:`13` **OpenAI**: Requires ``OPENAI_API_KEY``.
- :sup:`14` **Anthropic**: Requires ``ANTHROPIC_API_KEY``.
- :sup:`15` **OpenRouter**: Requires ``OPENROUTER_API_KEY``.
- :sup:`16` **Local vLLM**: Self-hosted on port ``30001`` using ``make vllm``.
